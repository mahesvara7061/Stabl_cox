# run_stabl_cox_from_counts.py - FIXED VERSION
# Key fix: Expanded alpha range from 1e-4 to 10 (instead of 0.01 to 0.1)

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Local imports
from .multi_omic_pipelines import multi_omic_stabl_cv
from .stabl import Stabl

# Survival models/metrics
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

# -------------------------------
# Utils (same as original)
# -------------------------------

def detect_sep(path):
    """Try to guess separator: prefer tab if '\t' found in first 1KB."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.read(1024)
    if "\t" in head and "," not in head:
        return "\t"
    if "\t" in head and "," in head:
        return "\t"
    return ","


def load_counts(counts_path, num_genes: int | None = None):
    """
    Read counts: first 2 columns = gene_name, entrez_id; remaining columns = samples.
    Returns:
      X (DataFrame): samples x genes (transposed, columns = gene_symbol)
    If num_genes is set, keep only the first num_genes genes in the *file order*.
    """
    sep = detect_sep(counts_path)
    df = pd.read_csv(counts_path, sep=sep, header=0)
    if df.shape[1] < 4:
        raise ValueError("File counts must have >= 4 columns (gene, entrez, and >=2 samples).")

    gene_col = df.columns[0]
    sample_cols = df.columns[2:]

    # Build deterministic gene order
    genes_in_order = pd.Index(df[gene_col].astype(str).tolist())
    first_occ_mask = ~genes_in_order.duplicated()
    unique_genes_in_order = genes_in_order[first_occ_mask]

    # Matrix genes x samples
    expr = df[sample_cols].copy()
    expr.index = df[gene_col].astype(str).values

    # Aggregate duplicates by median WITHOUT sorting
    expr = expr.groupby(expr.index, sort=False).median()

    # Reindex to first-appearance order
    expr = expr.reindex(unique_genes_in_order.intersection(expr.index))

    # Cap to first N genes if requested
    if num_genes is not None:
        expr = expr.iloc[:num_genes, :]

    # Transpose: samples x genes, coerce numeric
    X = expr.T.apply(pd.to_numeric, errors="coerce")
    return X


def load_clinical(clinical_path):
    """
    Read clinical with columns: sample id, OS, censored.
    Convention: censored = 0 (alive), 1 (dead) -> event=False/True.
    Returns DataFrame indexed by sample_id, columns ['time','event'] (event bool).
    """
    sep = detect_sep(clinical_path)
    clin = pd.read_csv(clinical_path, sep=sep, header=0)
    clin.columns = [c.replace("\ufeff", "") for c in clin.columns]

    # Standardize column names
    ren = {c: c.strip().lower() for c in clin.columns}
    clin = clin.rename(columns=ren)

    print("[DEBUG] clinical columns:", list(clin.columns))

    # Map column names
    col_map = {}
    # sample id
    for cand in ["sample id", "sample_id", "sample", "id", "case_id", "patient_id", "case submitter id", "cgga_id"]:
        if cand in clin.columns:
            col_map["sample_id"] = cand
            break
    # OS/time
    for cand in ["os", "time", "overall_survival", "overall survival", "survival_time", "survival time", "days_to_death", "days to death"]:
        if cand in clin.columns:
            col_map["time"] = cand
            break
    # censored/status/event
    for cand in ["censored", "censor", "status", "event", "vital_status", "vital status"]:
        if cand in clin.columns:
            col_map["censored"] = cand
            break

    if set(col_map.keys()) != {"sample_id", "time", "censored"}:
        raise ValueError(
            f"Cannot find all required columns (sample id, os/time, censored). "
            f"Mapped: {col_map}. Available: {list(clin.columns)}"
        )

    # Extract raw data
    sid_raw = clin[col_map["sample_id"]].astype(str)
    time_raw = clin[col_map["time"]]
    cens_raw = clin[col_map["censored"]]

    # Normalize time -> numeric
    time_str = (time_raw.astype(str)
                .str.replace(r"[^0-9\.,\-]", "", regex=True)
                .str.replace(",", ".", regex=False)
               )
    time = pd.to_numeric(time_str, errors="coerce")

    # Normalize censored: allow text variants
    cens_str = cens_raw.astype(str).str.strip().str.lower()
    text_map = {
        "alive": "0", "censored": "0", "living": "0", "no_event": "0", "no event": "0",
        "dead": "1", "deceased": "1", "event": "1", "died": "1",
        "0.0": "0", "1.0": "1"
    }
    cens_norm = cens_str.map(lambda x: text_map.get(x, x))
    cens = pd.to_numeric(cens_norm, errors="coerce")

    # 0 = alive -> False, 1 = dead -> True
    event = (cens == 1).astype("boolean")

    # Use .values to avoid index alignment
    sid_idx = pd.Index(sid_raw.astype(str).to_numpy(), name="sample_id")
    y = pd.DataFrame(
        {
            "time":  time.to_numpy(),
            "event": event.to_numpy(),
        },
        index=sid_idx
    )

    # Handle duplicate sample_ids
    if y.index.duplicated().any():
        print("[DEBUG] duplicated sample_id count:", y.index.duplicated().sum())
        y = y[~y.index.duplicated(keep="first")]

    # DEBUG before/after dropna
    print("[DEBUG] y shape BEFORE dropna:", y.shape)
    print("[DEBUG] head BEFORE dropna:\n", y.head(5))
    y = y.dropna(subset=["time", "event"])
    print("[DEBUG] y shape AFTER dropna:", y.shape)
    print("[DEBUG] sample_id (AFTER dropna) head:", list(y.index[:5]))
    return y


def align_X_y(X, y):
    """
    Intersect samples between X and y; sort index for consistency.
    """
    inter = X.index.astype(str).intersection(y.index.astype(str))
    X2 = X.loc[inter].sort_index()
    y2 = y.loc[inter].sort_index()
    # Drop samples with NaN time/event
    mask = (~y2['time'].isna()) & (~y2['event'].isna())
    X2 = X2.loc[mask]
    y2 = y2.loc[mask]
    return X2, y2


def build_stabl_cox(n_bootstraps=500, random_state=42, debug_dir=None):
    """
    ✅ v4 VERSION: Coxnet (Lasso) for STABL feature selection.
    
    Uses CoxnetSurvivalAnalysis with l1_ratio=1.0 (Lasso).
    We let Stabl automatically compute the alpha path using the fixed utils.py logic.
    """
    # We do NOT define lambda_grid manually here. 
    # Stabl will call utils.auto_mode_lambda_grid which now correctly uses Coxnet to find the path.
    lambda_grid = "auto"

    base = CoxnetSurvivalAnalysis(
        l1_ratio=1.0,
        fit_baseline_model=True,
        verbose=False
    )

    stabl_cox = Stabl(
        base_estimator=base,
        lambda_grid=lambda_grid,
        n_bootstraps=n_bootstraps,
        artificial_type="knockoff",
        artificial_proportion=0.5,
        sample_fraction=0.5,
        replace=False,
        bootstrap_threshold="median",
        fdr_threshold_range=np.arange(0.05, 1.01, 0.05),
        explore=True,
        n_explore=5,
        task_type="survival",
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        debug_dir=debug_dir
    )
    return stabl_cox


def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load data
    X = load_counts(args.counts, num_genes=args.num_genes)
    print(f"[INFO] Using {X.shape[1]} genes.")

    y = load_clinical(args.clinical)

    # 2) Align samples
    X, y = align_X_y(X, y)

    # ---------------------------------------------------------
    # DATA LEAKAGE CHECK
    # ---------------------------------------------------------
    # Ensure no target-related columns (OS, Censor, etc.) are in X
    leakage_keywords = [
        "os", "time", "overall_survival", "survival_time", "days_to_death", "days to death",
        "censored", "censor", "status", "event", "vital_status", "vital status",
        "outcome", "survival"
    ]
    # Check case-insensitive
    to_drop = []
    for col in X.columns:
        if str(col).strip().lower() in leakage_keywords:
            to_drop.append(col)
    
    if to_drop:
        print(f"[WARNING] Potential data leakage detected! Dropping columns from X: {to_drop}")
        X = X.drop(columns=to_drop)
    # ---------------------------------------------------------

    if X.shape[0] < 10 or X.shape[1] < 5:
        raise ValueError(f"Data too small after alignment: X={X.shape}, y={y.shape}. Check sample ID matching.")

    print(f"[INFO] X shape (samples x genes): {X.shape}")
    print(f"[INFO] y shape (samples x 2): {y.shape}")

    # Check events-per-variable ratio
    n_events = y['event'].sum()
    epv = n_events / X.shape[1]
    print(f"[INFO] Events: {n_events}/{X.shape[0]} ({100*n_events/X.shape[0]:.1f}%)")
    print(f"[INFO] Events-per-variable (EPV): {epv:.2f}")

    if epv < 10:
        print(f"[WARNING] EPV < 10 may lead to overfitting. Consider reducing --num_genes to {max(10, int(n_events/10))}")

    # 3) Estimators dict + models list
    stabl_cox = build_stabl_cox(n_bootstraps=args.n_boot, random_state=args.seed, debug_dir=args.debug_dir)
    
    # If lambda_grid is None (auto mode), we can't print range yet.
    if isinstance(stabl_cox.lambda_grid, dict):
        alpha_array = stabl_cox.lambda_grid.get("alpha", stabl_cox.lambda_grid.get("alphas", []))
        if len(alpha_array) > 0:
            # Flatten if it's a list of lists (Coxnet case sometimes)
            if isinstance(alpha_array[0], (list, np.ndarray)):
                alpha_array = [x[0] if isinstance(x, (list, np.ndarray)) else x for x in alpha_array]
            
            # Ensure numeric
            try:
                alpha_array = np.array(alpha_array, dtype=float).ravel()
                alpha_min, alpha_max = min(alpha_array), max(alpha_array)
                print(f"[INFO] Alpha range: {alpha_min:.6f} to {alpha_max:.2f}")
            except:
                print("[INFO] Alpha range: (complex structure)")
    else:
        print("[INFO] Alpha range: Auto-computed by Stabl (Coxnet)")

    estimators = {
        "stabl_cox": stabl_cox
    }
    models = ["STABL Cox"]

    # 4) CV splitter
    from sklearn.model_selection import RepeatedKFold
    outer_splitter = RepeatedKFold(n_splits=args.n_splits, n_repeats=args.n_repeats, random_state=args.seed)

    # 5) Run CV pipeline
    data_dict = {"mRNA": X}

    print("[INFO] Starting cross-validation (STABL Cox)...")
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
        original_counts_path=args.counts,
        original_clinical_path=args.clinical
    )

    # 6) Compute C-index
    pred_series = predictions["STABL Cox"].loc[y.index]
    c_index = concordance_index_censored(
        y["event"].astype(bool).to_numpy(),
        y["time"].to_numpy(),
        pred_series.to_numpy()
    )[0]
    print(f"[RESULT] CV median C-index (STABL Cox): {c_index:.3f}")

    # 7) Save summary
    out_summary = Path(args.outdir) / "Summary" / "quick_summary.txt"
    os.makedirs(out_summary.parent, exist_ok=True)
    with open(out_summary, "w") as f:
        f.write(f"CV median C-index (STABL Cox): {c_index:.4f}\n")
        f.write(f"Samples: {X.shape[0]}, Genes: {X.shape[1]}\n")
        f.write(f"Events: {n_events} (EPV: {epv:.2f})\n")
        if stabl_cox.lambda_grid is not None:
             f.write(f"Alpha range: Provided/Computed\n")
        else:
             f.write(f"Alpha range: Auto (Coxnet)\n")

    print(f"[DONE] Full results (CSV/Figures) saved at: {Path(args.outdir).resolve()}")
    print(f"[DONE] Quick summary: {out_summary.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run STABL Cox survival on counts + clinical (FIXED VERSION)")
    parser.add_argument("--counts", required=True, help="Path to counts file (TSV or CSV)")
    parser.add_argument("--clinical", required=True, help="Path to clinical file (TSV/CSV)")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--n_boot", type=int, default=500, help="Number of bootstraps for STABL")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--n_repeats", type=int, default=5, help="Number of CV repeats")
    parser.add_argument("--seed", type=int, default=42, help="random_state")
    parser.add_argument(
        "--num_genes",
        type=int,
        default=100,
        help="Keep only the first N genes in file order (after duplicate aggregation)."
    )
    parser.add_argument("--debug_dir", type=str, default=None, help="Directory to save debug info")

    args = parser.parse_args()
    main(args)



