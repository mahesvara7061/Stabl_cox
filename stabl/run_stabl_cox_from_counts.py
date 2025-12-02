# run_stabl_cox_from_counts.py
# End-to-end: đọc counts + clinical -> STABL Cox (survival) CV -> lưu kết quả

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ====== Local imports: giả sử các file *.py đã ở cùng thư mục hoặc đã là module ======
from .multi_omic_pipelines import multi_omic_stabl_cv
from .stabl import Stabl

# Survival models/metrics
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

# -------------------------------
# Utils
# -------------------------------

def detect_sep(path):
    """Thử đoán phân cách: ưu tiên tab nếu thấy '\t' trong 1KB đầu."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.read(1024)
    if "\t" in head and "," not in head:
        return "\t"
    # nếu cả hai đều có, ưu tiên tab (định dạng counts thường là TSV)
    if "\t" in head and "," in head:
        return "\t"
    return ","


# def load_counts(counts_path):
#     """
#     Đọc counts: 2 cột đầu là gene name + entrez id, các cột sau là sample.
#     Trả về:
#       X (DataFrame): samples x genes (đã transpose, cột là gene_symbol)
#     """
#     sep = detect_sep(counts_path)
#     df = pd.read_csv(counts_path, sep=sep, header=0)
#     if df.shape[1] < 4:
#         raise ValueError("File counts phải có >= 4 cột (gene, entrez, và >=2 samples).")

#     # Chuẩn tên 2 cột đầu
#     col0, col1 = df.columns[:2]
#     gene_col = col0
#     entrez_col = col1

#     # Lấy ma trận (genes x samples)
#     sample_cols = df.columns[2:]
#     expr = df[sample_cols].copy()

#     # Đặt index = gene symbol; nếu trùng, gộp bằng median
#     genes = df[gene_col].astype(str).values
#     expr.index = genes
#     # Nếu có gene trùng tên -> aggregate theo median
#     expr = expr.groupby(expr.index).median()

#     # Transpose: samples x genes
#     X = expr.T
#     # Đảm bảo kiểu số
#     X = X.apply(pd.to_numeric, errors="coerce")
#     # print(list(X.index[:5]))
#     return X
def load_counts(counts_path, num_genes: int | None = None):
    """
    Read counts: first 2 columns = gene_name, entrez_id; remaining columns = samples.
    Returns:
      X (DataFrame): samples x genes (transposed, columns = gene_symbol)
    If num_genes is set, keep only the first num_genes genes in the *file order*
    (after duplicate symbols are aggregated by median).
    """
    sep = detect_sep(counts_path)
    df = pd.read_csv(counts_path, sep=sep, header=0)
    if df.shape[1] < 4:
        raise ValueError("File counts must have >= 4 columns (gene, entrez, and >=2 samples).")

    gene_col = df.columns[0]
    sample_cols = df.columns[2:]

    # Build a deterministic gene order in the order of first appearance in the file
    genes_in_order = pd.Index(df[gene_col].astype(str).tolist())
    first_occ_mask = ~genes_in_order.duplicated()
    unique_genes_in_order = genes_in_order[first_occ_mask]

    # Matrix genes x samples
    expr = df[sample_cols].copy()
    expr.index = df[gene_col].astype(str).values

    # Aggregate duplicates by median WITHOUT sorting (preserve first-appearance order)
    expr = expr.groupby(expr.index, sort=False).median()

    # Reindex to the first-appearance order (drop any genes that vanished after coercion)
    expr = expr.reindex(unique_genes_in_order.intersection(expr.index))

    # Cap to first N genes if requested
    if num_genes is not None:
        expr = expr.iloc[:num_genes, :]

    # Transpose: samples x genes, coerce numeric
    X = expr.T.apply(pd.to_numeric, errors="coerce")
    return X


import re

def load_clinical(clinical_path):
    """
    Đọc clinical với các cột: sample id, OS, censored.
    Quy ước: censored = 0 (alive), 1 (dead)  -> event=False/True.
    Trả về DataFrame index theo sample_id, cột ['time','event'] (event bool).
    Có in DEBUG để chẩn đoán nếu dữ liệu trống.
    """
    sep = detect_sep(clinical_path)
    # đọc, giữ nguyên header; strip BOM ở tên cột
    clin = pd.read_csv(clinical_path, sep=sep, header=0)
    clin.columns = [c.replace("\ufeff", "") for c in clin.columns]

    # Chuẩn tên cột
    ren = {c: c.strip().lower() for c in clin.columns}
    clin = clin.rename(columns=ren)

    print("[DEBUG] clinical columns:", list(clin.columns))

    # Map tên cột linh hoạt (mở rộng alias)
    col_map = {}
    # sample id
    for cand in ["sample id", "sample_id", "sample", "id", "case_id", "patient_id", "case submitter id"]:
        if cand in clin.columns:
            col_map["sample_id"] = cand
            break
    # OS/time
    for cand in ["os", "time", "overall_survival", "overall survival", "survival_time", "survival time", "days_to_death", "days to death"]:
        if cand in clin.columns:
            col_map["time"] = cand
            break
    # censored/status/event  (BỔ SUNG 'censored' vào đây)
    for cand in ["censored", "censor", "status", "event", "vital_status", "vital status"]:
        if cand in clin.columns:
            col_map["censored"] = cand
            break

    if set(col_map.keys()) != {"sample_id", "time", "censored"}:
        raise ValueError(
            f"Không tìm đủ cột (sample id, os/time, censored). "
            f"Đã map: {col_map}. Tất cả cột có: {list(clin.columns)}"
        )

    # Lấy dữ liệu thô
    sid_raw = clin[col_map["sample_id"]].astype(str)
    time_raw = clin[col_map["time"]]
    cens_raw = clin[col_map["censored"]]

    # In vài dòng thô để soi
    # print("[DEBUG] sample_id (raw) head:", list(sid_raw.head(5)))
    # print("[DEBUG] time (raw) head:", list(time_raw.head(5)))
    # print("[DEBUG] censored (raw) head:", list(cens_raw.head(5)))

    # Chuẩn hoá/parse time -> numeric
    # Loại ký tự không phải số/chấm/phẩy/trừ, đổi phẩy -> chấm
    time_str = (time_raw.astype(str)
                .str.replace(r"[^0-9\.,\-]", "", regex=True)
                .str.replace(",", ".", regex=False)
               )
    time = pd.to_numeric(time_str, errors="coerce")

    # Chuẩn hoá censored: cho phép các biến thể văn bản
    cens_str = cens_raw.astype(str).str.strip().str.lower()
    # map nhanh chữ -> số
    text_map = {
        "alive": "0", "censored": "0", "living": "0", "no_event": "0", "no event": "0",
        "dead": "1", "deceased": "1", "event": "1", "died": "1",
        "0.0": "0", "1.0": "1"
    }
    cens_norm = cens_str.map(lambda x: text_map.get(x, x))
    cens = pd.to_numeric(cens_norm, errors="coerce")

    # Kiểm tra phân bổ trước khi drop
    # print("[DEBUG] time NaN count:", int(time.isna().sum()), "/", len(time))
    # print("[DEBUG] censored unique (pre-parse):", sorted(pd.unique(cens_raw.astype(str))[:20]))
    # print("[DEBUG] censored unique (numeric parsed):", sorted(pd.unique(cens[~cens.isna()])[:20]))

    # 0 = alive -> False, 1 = dead -> True
    # 0 = alive -> False, 1 = dead -> True
    event = (cens == 1).astype("boolean")

# DÙNG .values / .to_numpy() để tránh align theo index
    sid_idx = pd.Index(sid_raw.astype(str).to_numpy(), name="sample_id")
    y = pd.DataFrame(
        {
            "time":  time.to_numpy(),      # <--- quan trọng
            "event": event.to_numpy(),     # <--- quan trọng
        },
        index=sid_idx
    )

    # Nếu có trùng sample_id, giữ bản ghi đầu tiên (hoặc tuỳ logic bạn)
    if y.index.duplicated().any():
        print("[DEBUG] duplicated sample_id count:", y.index.duplicated().sum())
        y = y[~y.index.duplicated(keep="first")]

    # DEBUG trước/sau dropna
    print("[DEBUG] y shape BEFORE dropna:", y.shape)
    print("[DEBUG] head BEFORE dropna:\n", y.head(5))
    y = y.dropna(subset=["time", "event"])
    print("[DEBUG] y shape AFTER dropna:", y.shape)
    print("[DEBUG] sample_id (AFTER dropna) head:", list(y.index[:5]))
    return y





def align_X_y(X, y):
    """
    Cắt giao giữa samples; sort index cho khớp.
    """
    inter = X.index.astype(str).intersection(y.index.astype(str))
    X2 = X.loc[inter].sort_index()
    y2 = y.loc[inter].sort_index()
    # Loại mẫu có time/event NaN
    mask = (~y2['time'].isna()) & (~y2['event'].isna())
    X2 = X2.loc[mask]
    y2 = y2.loc[mask]
    return X2, y2


def build_stabl_cox(n_bootstraps=500, random_state=42):
    """
    Tạo STABL với CoxnetSurvivalAnalysis, grid alphas hợp lệ cho scikit-survival.
    Lưu ý: Stabl sẽ đặt tham số qua set_params(**lambda_val),
           nên ta cần lambda_grid với key 'alphas', value là list các np.array([alpha]).
    """
    # lưới alpha logspace (đủ dày để ổn định nhưng không quá nặng)
    alpha_list = np.logspace(-2, -1, 50) # từ 1e-4 đến 1e4, 50 giá trị
    lambda_grid = {
        "alphas": [np.array([a]) for a in alpha_list],
        # bạn có thể cho "l1_ratio" là hằng số trong base_estimator; nếu muốn grid cả l1_ratio:
        # "l1_ratio": [1.0, 0.8, 0.6]
    }
    # Each alpha tested independently in separate bootstrap iterations
    # lambda_grid = {
    #     "alphas": [alpha_list]  # Single array with all alphas
    # }

    base = CoxnetSurvivalAnalysis(
        l1_ratio=0.7,       # LASSO Cox; đổi 0.6–1.0 nếu muốn ElasticNet Coxnet
        tol=1e-6,
        max_iter=2_000_000
    )

    stabl_cox = Stabl(
        base_estimator=base,
        lambda_grid=lambda_grid,
        n_bootstraps=n_bootstraps,
        artificial_type="random_permutation",  # dùng decoys để điều chỉnh FDP+
        artificial_proportion=0.5,
        sample_fraction=0.7,                   # bootstrap fraction (không thay thế)
        replace=False,
        bootstrap_threshold="median",
        fdr_threshold_range=np.arange(0.05, 1.01, 0.05),
        explore=True,                          # nếu không có feature qua FDR, chọn n_explore tốt nhất
        n_explore=5,
        task_type="survival",
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    return stabl_cox


def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    # 1) Load dữ liệu
    X = load_counts(args.counts, num_genes=args.num_genes)
    print(f"[INFO] Capped to first {X.shape[1]} genes.")

    y = load_clinical(args.clinical)

    # 2) Căn chỉnh mẫu
    X, y = align_X_y(X, y)
    if X.shape[0] < 10 or X.shape[1] < 5:
        raise ValueError(f"Dữ liệu quá nhỏ sau khi căn chỉnh: X={X.shape}, y={y.shape}. Kiểm tra ID mẫu khớp ở cả 2 file.")

    print(f"[INFO] X shape (samples x genes): {X.shape}")
    print(f"[INFO] y shape (samples x 2): {y.shape}")

    # 3) Estimators dict + models list
    stabl_cox = build_stabl_cox(n_bootstraps=args.n_boot, random_state=args.seed)
    estimators = {
        "stabl_cox": stabl_cox
    }
    models = ["STABL Cox"]  # chỉ chạy STABL Cox cho survival

    # 4) CV splitter (survival không stratify): dùng RepeatedKFold
    from sklearn.model_selection import RepeatedKFold
    outer_splitter = RepeatedKFold(n_splits=args.n_splits, n_repeats=args.n_repeats, random_state=args.seed)

    # 5) Gọi pipeline CV một-omics (mRNA)
    data_dict = {"mRNA": X}

    print("[INFO] Bắt đầu cross-validation (STABL Cox)...")
    predictions = multi_omic_stabl_cv(
        data_dict=data_dict,
        y=y,
        outer_splitter=outer_splitter,
        estimators=estimators,
        task_type="survival",
        save_path=Path(args.outdir),
        models=models,
        outer_groups=None,
        early_fusion=False,
        late_fusion=False,
        n_iter_lf=10000,
        original_counts_path=args.counts,        # ← ADD
        original_clinical_path=args.clinical     # ← ADD
    )

    # 6) Tính C-index trên median predictions của CV
    # predictions trả về dict: {model: pd.Series (median across folds)}
    pred_series = predictions["STABL Cox"].loc[y.index]
    c_index = concordance_index_censored(
        y["event"].astype(bool).to_numpy(),
        y["time"].to_numpy(),
        pred_series.to_numpy()
    )[0]
    print(f"[RESULT] CV median C-index (STABL Cox): {c_index:.3f}")

    # 7) Lưu C-index vào file tóm tắt
    out_summary = Path(args.outdir) / "Summary" / "quick_summary.txt"
    os.makedirs(out_summary.parent, exist_ok=True)
    with open(out_summary, "w") as f:
        f.write(f"CV median C-index (STABL Cox): {c_index:.4f}\n")
        f.write(f"Samples: {X.shape[0]}, Genes: {X.shape[1]}\n")

    print(f"[DONE] Kết quả đầy đủ (CSV/Hình) đã lưu ở: {Path(args.outdir).resolve()}")
    print(f"[DONE] Tóm tắt nhanh: {out_summary.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run STABL Cox survival on counts + clinical")
    parser.add_argument("--counts", required=True, help="Đường dẫn file counts (TSV hoặc CSV): 2 cột đầu = gene name, entrez id; các cột sau = sample IDs")
    parser.add_argument("--clinical", required=True, help="Đường dẫn file clinical (TSV/CSV) có cột: sample id, OS, censored (1=dead, 2=alive)")
    parser.add_argument("--outdir", required=True, help="Thư mục xuất kết quả")
    parser.add_argument("--n_boot", type=int, default=500, help="Số bootstrap cho STABL (mặc định 500; tăng lên 1000 cho kết quả ổn hơn)")
    parser.add_argument("--n_splits", type=int, default=5, help="Số fold CV")
    parser.add_argument("--n_repeats", type=int, default=5, help="Số lần lặp CV")
    parser.add_argument("--seed", type=int, default=42, help="random_state")
    parser.add_argument(
    "--num_genes",
    type=int,
    default=100,   # set to 100 as you requested
    help="Keep only the first N genes in file order (after duplicate aggregation)."
)

    args = parser.parse_args()
    main(args)
