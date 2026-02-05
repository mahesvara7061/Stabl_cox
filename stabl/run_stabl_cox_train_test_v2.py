import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import io
import traceback
import warnings
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Local imports
try:
    from .stabl import Stabl, save_stabl_results
    from .preprocessing import LowInfoFilter
except ImportError:
    # If running as script from stabl/ folder
    sys.path.append(str(Path(__file__).parent.parent))
    from stabl.stabl import Stabl, save_stabl_results
    from stabl.preprocessing import LowInfoFilter

from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
    brier_score
)
from sksurv.util import Surv
from sksurv.nonparametric import kaplan_meier_estimator
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Thêm thư viện lifelines cho phân tích đơn biến
try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test
except ImportError:
    print("[WARNING] Thư viện 'lifelines' chưa được cài đặt. Một số phân tích đơn biến sẽ bị bỏ qua.")
    print("          Vui lòng cài đặt: pip install lifelines")

# -------------------------------
# Utils
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


def load_counts(counts_path, num_genes: int | None = None, debug_dir: str | None = None):
    """
    Read counts: first 2 columns = gene_name, entrez_id; remaining columns = samples.
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
    
    if expr.index.duplicated().any():
        if debug_dir:
            try:
                os.makedirs(debug_dir, exist_ok=True)
                dup_genes = expr.index[expr.index.duplicated()].unique()
                pd.Series(dup_genes, name="Duplicated_Genes").to_csv(Path(debug_dir) / "duplicated_genes.csv", index=False)
            except Exception:
                pass

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


def load_clinical(clinical_path, target_type="OS"):
    """
    Read clinical with columns: sample id, Time, Event.
    target_type: "OS" (Overall Survival) or "RFS" (Recurrence/Relapse Free Survival)
    Convention: censored = 0 (no event), 1 (event) -> event=False/True.
    """
    sep = detect_sep(clinical_path)
    clin = pd.read_csv(clinical_path, sep=sep, header=0)
    clin.columns = [c.replace("\ufeff", "") for c in clin.columns]

    ren = {c: c.strip().lower() for c in clin.columns}
    clin = clin.rename(columns=ren)

    col_map = {}
    # 1. Sample ID candidates
    for cand in ["sample id", "sample_id", "sample", "id", "case_id", "patient_id", "case submitter id", "cgga_id"]:
        if cand in clin.columns:
            col_map["sample_id"] = cand
            break
            
    # Be flexible but prioritize based on target_type
    if target_type.upper() == "RFS":
        time_cands = ["rfs_time", "dfs_time", "rfs_days", "rfs_months", "dfs_days", "df_time", "recurrence_time", "time_to_recurrence", "rfs", "dfs"]
        event_cands = ["rfs_status", "dfs_status", "recurrence_status", "recurrence_event", "recurrence", "relapse", "relapse_status"]
    else: # OS
        time_cands = ["os_time", "os_days", "os_months", "overall_survival", "overall survival", "survival_time", "survival time", "days_to_death", "days to death", "time", "os"]
        event_cands = ["os_status", "vital_status", "vital status", "censored", "censor", "status", "event", "survival_status", "os"]

    # 2. Search Time
    for cand in time_cands:
        if cand in clin.columns:
            col_map["time"] = cand
            break
    # Fallback generic time if specific not found
    if "time" not in col_map:
        for cand in ["time", "survival_time", "days_to_death"]:
            if cand in clin.columns:
                col_map["time"] = cand
                break

    # 3. Search Event
    for cand in event_cands:
        if cand in clin.columns:
            col_map["censored"] = cand
            break
    # Fallback generic event
    if "censored" not in col_map:
        for cand in ["censored", "censor", "status", "event", "vital_status"]:
            if cand in clin.columns:
                col_map["censored"] = cand
                break

    if set(col_map.keys()) != {"sample_id", "time", "censored"}:
        raise ValueError(f"Cannot find all required columns for {target_type}. Mapped: {col_map}. Available: {list(clin.columns)}")

    sid_raw = clin[col_map["sample_id"]].astype(str)
    time_raw = clin[col_map["time"]]
    cens_raw = clin[col_map["censored"]]

    time_str = (time_raw.astype(str)
                .str.replace(r"[^0-9\.,\-]", "", regex=True)
                .str.replace(",", ".", regex=False)
               )
    time = pd.to_numeric(time_str, errors="coerce")

    cens_str = cens_raw.astype(str).str.strip().str.lower()
    text_map = {
        "alive": "0", "censored": "0", "living": "0", "no_event": "0", "no event": "0", "disease free": "0", "recurrence free": "0",
        "dead": "1", "deceased": "1", "event": "1", "died": "1", "recurred": "1", "recurrence": "1", "relapse": "1",
        "0.0": "0", "1.0": "1"
    }
    cens_norm = cens_str.map(lambda x: text_map.get(x, x))
    cens = pd.to_numeric(cens_norm, errors="coerce")
    event = (cens == 1).astype("boolean")

    sid_idx = pd.Index(sid_raw.astype(str).to_numpy(), name="sample_id")
    y = pd.DataFrame({"time":  time.to_numpy(), "event": event.to_numpy()}, index=sid_idx)

    if y.index.duplicated().any():
        y = y[~y.index.duplicated(keep="first")]

    y = y.dropna(subset=["time", "event"])
    print(f"[INFO] Loaded clinical data ({target_type}) with {y.shape[0]} samples. (Time col: {col_map['time']}, Event col: {col_map['censored']})")
    return y


def align_X_y(X, y):
    inter = X.index.astype(str).intersection(y.index.astype(str))
    X2 = X.loc[inter].sort_index()
    y2 = y.loc[inter].sort_index()
    mask = (~y2['time'].isna()) & (~y2['event'].isna())
    X2 = X2.loc[mask]
    y2 = y2.loc[mask]
    return X2, y2


def run_univariate_cox(X, y, p_thresh=0.05, debug_dir=None):
    print(f"[INFO] Running univariate Cox selection (p < {p_thresh})...")
    try:
        from lifelines import CoxPHFitter
    except ImportError:
        return X

    genes = X.columns
    times = y['time'].values
    events = y['event'].values.astype(int)
    
    def fit_one_gene(gene, values):
        try:
            df = pd.DataFrame({'feature': values, 'T': times, 'E': events}).dropna()
            if df['feature'].nunique() < 2: return (gene, 1.0)
            cph = CoxPHFitter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cph.fit(df, duration_col='T', event_col='E')
            return (gene, cph.summary.loc['feature', 'p'])
        except Exception:
            return (gene, 1.0)

    results = Parallel(n_jobs=-1, verbose=5)(delayed(fit_one_gene)(g, X[g].values) for g in genes)
    res_df = pd.DataFrame(results, columns=['gene', 'p_value'])
    
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        res_df.to_csv(Path(debug_dir) / "univariate_cox_results.csv", index=False)
        
    selected = res_df[res_df['p_value'] < p_thresh]['gene'].tolist()
    print(f"[INFO] Univariate Cox: {len(selected)}/{len(genes)} features passed p < {p_thresh}")
    if len(selected) == 0: return X
    return X[selected]

# -------------------------------
# NEW: Comprehensive Evaluation Functions
# -------------------------------

def evaluate_survival_model_comprehensive(model, X_train, y_train, X_test, y_test, model_name, outdir):
    """
    Đánh giá toàn diện mô hình sinh tồn (Đã sửa lỗi version scikit-survival)
    """
    results = {}
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n--- Đánh giá nâng cao cho: {model_name} ---")

    # Dự đoán Risk Score
    try:
        risk_scores = model.predict(X_test)
    except Exception as e:
        print(f"[WARN] Không thể dự đoán risk score cho {model_name}: {e}")
        return results

    # --- 1. Harrell's C-index ---
    try:
        harrell_c = concordance_index_censored(y_test["event"], y_test["time"], risk_scores)[0]
        results["Harrell_C"] = harrell_c
        print(f"   >> Harrell's C-index: {harrell_c:.4f}")
    except Exception as e:
        results["Harrell_C"] = np.nan

    # --- 2. Uno's C-index ---
    # Uno C yêu cầu tau phải nhỏ hơn thời gian lớn nhất của cả train và test
    tau = min(y_test["time"].max(), y_train["time"].max()) - 1e-5
    try:
        uno_c, _, _, _, _ = concordance_index_ipcw(y_train, y_test, risk_scores, tau=tau)
        print(f"   >> Uno's C-index: {uno_c:.4f}")
        results["Uno_C_Index"] = uno_c
    except Exception as e:
        print(f"   >> Lỗi tính Uno's C: {e}")
        results["Uno_C_Index"] = np.nan

    # --- 3. Brier Score & IBS ---
    if hasattr(model, "predict_survival_function"):
        try:
            survs = model.predict_survival_function(X_test)
            
            # --- [FIX QUAN TRỌNG] ---
            # Xác định các mốc thời gian để tính Brier
            # Lower bound: Bắt đầu từ thời điểm sớm nhất của Test set (không cần phụ thuộc Train set vì G(t)=1 tại t=0)
            lower = y_test["time"].min()
            if lower <= 0: lower = 1e-5 # Time must be positive
            
            # Upper bound: Bị giới hạn bởi thời gian theo dõi lớn nhất của Train set (để tính được IPCW)
            upper = min(y_test["time"].max(), y_train["time"].max()) - 1e-5
            
            if lower >= upper:
                print(f"   >> [WARN] Không thể tính IBS: Valid time range is empty (Lower: {lower}, Upper: {upper})")
                results["IBS"] = np.nan
            else:
                times_brier = np.linspace(lower, upper, 100)
                
                # Thay vì truyền 'survs' (object), ta tính giá trị cụ thể tại times_brier
                # Tạo ma trận (n_samples, n_times)
                preds = np.row_stack([fn(times_brier) for fn in survs])
                
                # Bây giờ truyền ma trận số thực 'preds' vào thay vì 'survs'
                ibs = integrated_brier_score(y_train, y_test, preds, times_brier)
                print(f"   >> Integrated Brier Score (IBS): {ibs:.4f} (Thấp là tốt)")
                results["IBS"] = ibs
                
                # Vẽ Prediction Error Curve
                times_plot, brier_scores_vals = brier_score(y_train, y_test, preds, times_brier)
                
                plt.figure(figsize=(8, 6))
                plt.plot(times_plot, brier_scores_vals, color="purple", lw=2, label=f"IBS = {ibs:.3f}")
                plt.axhline(0.25, color="gray", linestyle="--", alpha=0.5, label="Random (0.25)")
                plt.xlabel("Time (Days)")
                plt.ylabel("Brier Score (Prediction Error)")
                plt.title(f"{model_name}: Prediction Error Curve")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.savefig(out_path / f"{model_name}_brier_score_plot.png")
                plt.close()
            
        except Exception as e:
            print(f"   >> Lỗi tính Brier/IBS: {e}")
            import traceback
            traceback.print_exc()
            results["IBS"] = np.nan
    else:
        print(f"   >> Model {model_name} không hỗ trợ predict_survival_function -> Bỏ qua IBS.")

    # --- 4. Time-dependent AUC ---
    try:
        # Chọn các mốc thời gian từ percentile 10 đến 90 của tập TEST
        times_auc = np.percentile(y_test["time"], np.linspace(10, 90, 15))
        # Lọc bỏ các mốc thời gian vượt quá train set (nếu có) để tránh lỗi
        times_auc = times_auc[times_auc < y_train["time"].max()]
        
        auc_vals, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, times_auc)
        
        plt.figure(figsize=(8, 6))
        plt.plot(times_auc, auc_vals, "o-", color="blue", label=f"Mean AUC = {mean_auc:.3f}")
        plt.axhline(0.5, color="red", linestyle="--", label="Random (0.5)")
        plt.xlabel("Time (Days)")
        plt.ylabel("Time-dependent AUC")
        plt.title(f"{model_name}: Time-dependent AUC")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0.0, 1.0) # Mở rộng y-lim để nhìn rõ nếu AUC thấp
        plt.savefig(out_path / f"{model_name}_time_dependent_auc.png")
        plt.close()
        results["Mean_Time_AUC"] = mean_auc
    except Exception as e:
        print(f"   >> Lỗi vẽ Time-dependent AUC: {e}")

    # --- 5. Kaplan-Meier Risk Stratification ---
    try:
        median_risk = np.median(risk_scores)
        high_risk_mask = risk_scores > median_risk
        
        plt.figure(figsize=(8, 6))
        
        # SỬA LỖI UNPACK: kaplan_meier_estimator chỉ trả về 2 giá trị
        # Low Risk
        time_L, surv_L = kaplan_meier_estimator(
            y_test["event"][~high_risk_mask], y_test["time"][~high_risk_mask]
        )
        plt.step(time_L, surv_L, where="post", label="Low Risk", color="green", lw=2)
        
        # High Risk
        time_H, surv_H = kaplan_meier_estimator(
            y_test["event"][high_risk_mask], y_test["time"][high_risk_mask]
        )
        plt.step(time_H, surv_H, where="post", label="High Risk", color="red", lw=2)

        # Log-rank test
        if 'logrank_test' in globals():
            lr = logrank_test(
                y_test["time"][high_risk_mask], y_test["time"][~high_risk_mask],
                y_test["event"][high_risk_mask], y_test["event"][~high_risk_mask]
            )
            plt.title(f"{model_name}: Risk Stratification (Log-rank p={lr.p_value:.4f})")
        else:
            plt.title(f"{model_name}: Kaplan-Meier Risk Stratification")

        plt.xlabel("Time (Days)")
        plt.ylabel("Survival Probability")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(out_path / f"{model_name}_km_risk_groups.png")
        plt.close()
    except Exception as e:
        print(f"   >> Lỗi vẽ KM Stratification: {e}")

    return results


def analyze_individual_genes(X, y, selected_features, outdir, dataset_label="", trained_model=None, trained_median=None):
    """
    Phân tích chi tiết từng gene được chọn + Biosignature (Risk Score form Multivariate Cox):
    1. Univariate Cox Regression (HR, CI, p-value)
    2. Kaplan-Meier Plot (High vs Low expression) with detailed stats.
    3. Biosignature Evaluation (Refit, Transfer Fixed Threshold, Transfer Adaptive Threshold).
    """
    if 'CoxPHFitter' not in globals():
        print("[WARNING] Không có thư viện lifelines. Bỏ qua phân tích từng gene.")
        return None, None

    label_str = f" ({dataset_label})" if dataset_label else ""
    print(f"\n[INFO] Đang phân tích đơn biến{label_str} cho {len(selected_features)} gene được chọn...")
    
    folder_name = "individual_genes_analysis"
    csv_name = "univariate_analysis_selected_genes.csv"
    
    if dataset_label:
        folder_name += f"_{dataset_label}"
        csv_name = re.sub(r"\.csv$", f"_{dataset_label}.csv", csv_name)

    out_path = Path(outdir) / folder_name
    out_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # Ensure X has the selected features (handle missing if any)
    available_feats = [f for f in selected_features if f in X.columns]
    if len(available_feats) < len(selected_features):
        print(f"[WARN] Có {len(selected_features)-len(available_feats)} features bị thiếu trong dataset này.")
    
    df_analysis = X[available_feats].copy()
    df_analysis['T'] = y['time'].values
    df_analysis['E'] = y['event'].astype(int).values

    # --- Helper: Plot KM with detailed stats ---
    def plot_km_with_stats(data_values, name_for_plot, time_col, event_col, save_file, add_to_results=False, fixed_threshold=None):
        print(f"   >> [PLOT] Generating VS KM for: {name_for_plot}")
        try:
            # 1. Split
            if fixed_threshold is not None:
                threshold_val = fixed_threshold
                suffix = " (Fixed Thr)"
            else:
                threshold_val = np.median(data_values)
                suffix = " (Median Thr)"

            # High means > threshold
            high_mask = data_values > threshold_val
            
            n_high = high_mask.sum()
            n_low = (~high_mask).sum()
            
            if n_high == 0 or n_low == 0:
                print(f"      >> [SKIP] Cannot split {name_for_plot} (High={n_high}, Low={n_low})")
                return

            # 2. Log-rank test
            T_high, E_high = time_col[high_mask], event_col[high_mask]
            T_low, E_low = time_col[~high_mask], event_col[~high_mask]
            
            # Explicitly use keyword arguments for safety
            lr_res = logrank_test(
                durations_A=T_high, durations_B=T_low, 
                event_observed_A=E_high, event_observed_B=E_low
            )
            p_logrank = lr_res.p_value

            # 3. Binary Cox for HR (High vs Low)
            # Feature "group": 1=High, 0=Low
            # Note: We must suppress warnings for convergence if separation is perfect
            df_bin = pd.DataFrame({'group': high_mask.astype(int), 'T': time_col, 'E': event_col})
            cph_bin = CoxPHFitter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cph_bin.fit(df_bin, duration_col='T', event_col='E')
            
            hr_high = cph_bin.summary.loc['group', 'exp(coef)']
            lower_ci = cph_bin.summary.loc['group', 'exp(coef) lower 95%']
            upper_ci = cph_bin.summary.loc['group', 'exp(coef) upper 95%']
            p_hr = cph_bin.summary.loc['group', 'p']

            if add_to_results:
                results.append({
                    "Gene": name_for_plot,
                    "Hazard_Ratio": hr_high,
                    "CI_Lower": lower_ci,
                    "CI_Upper": upper_ci,
                    "P_value": p_hr,
                    "Role": "Biosignature" if "Biosignature" in name_for_plot else "Gene"
                })

            # 4. Plot
            plt.figure(figsize=(7, 6))
            ax = plt.subplot(111)
            
            kmf_high = KaplanMeierFitter()
            kmf_low = KaplanMeierFitter()
            
            # Plot High
            kmf_high.fit(T_high, event_observed=E_high, label=f"High (n={n_high})")
            kmf_high.plot_survival_function(ax=ax, color='red')
            
            # Plot Low
            kmf_low.fit(T_low, event_observed=E_low, label=f"Low (n={n_low})")
            kmf_low.plot_survival_function(ax=ax, color='green')
            
            # Construct Stats Text in REQUESTED format
            stats_str = (
                f"Log-rank p: {p_logrank:.2e}\n"
                f"HR(High): {hr_high:.2f}\n"
                f"p(HR): {p_hr:.2e}\n"
                f"n(High)={n_high}, n(Low)={n_low}\n"
                f"Threshold: {threshold_val:.4f}"
            )

            # Add Gene Count for Biosignature
            if "Biosignature" in name_for_plot:
                stats_str += f"\nN_Genes: {len(available_feats)}"
            
            ax.add_artist(plt.legend(loc='best')) 
            
            # Add the stats box (bottom left or best corner)
            ax.text(0.02, 0.05, stats_str, transform=ax.transAxes, fontsize=10,
                    verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

            plt.title(f"{name_for_plot}{suffix}")
            plt.ylabel("Survival Probability")
            plt.xlabel("Time")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_file)
            plt.close()

            # --- NEW: Save Prognosis Labels to CSV ---
            # Only if this is a Biosignature plot (Refit or Transfer)
            if "Biosignature" in name_for_plot:
                try:
                    # Create DataFrame with sample_id and prognosis
                    # high_mask: True = High Risk (Bad Prognosis), False = Low Risk (Good Prognosis)
                    prognosis_labels = ["Bad" if is_high else "Good" for is_high in high_mask]
                    
                    # Core info
                    df_core = pd.DataFrame({
                        "sample_id": df_analysis.index,
                        "Label": prognosis_labels,
                        "time": time_col,
                        "event": event_col,
                        "risk_score": data_values,
                        "threshold_used": threshold_val
                    })
                    
                    # Add gene expressions (drop T and E from df_analysis as we have them or don't need duplicates)
                    df_genes = df_analysis[available_feats] # available_feats is defined in parent scope
                    
                    # Combine: sample_id, prognosis, gene1, gene2...
                    # Align by index just in case, though they should be aligned
                    df_export = pd.concat([df_core.set_index("sample_id"), df_genes], axis=1).reset_index()
                    
                    csv_name = f"prognosis_{name_for_plot}.csv"
                    df_export.to_csv(out_path / csv_name, index=False)
                    print(f"      >> [INFO] Saved prognosis labels + expression to: {csv_name}")
                except Exception as e:
                    print(f"      >> [WARN] Could not save prognosis CSV: {e}")
            
        except Exception as e:
            print(f"Error plotting KM for {name_for_plot}: {e}")

    # --- Loop Genes ---
    for gene in available_feats:
        # Univariate Continuous Cox (Stats for CSV)
        try:
            cph = CoxPHFitter()
            df_gene = df_analysis[[gene, 'T', 'E']].dropna()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cph.fit(df_gene, duration_col='T', event_col='E')
            
            summary = cph.summary.loc[gene]
            hr = summary['exp(coef)']
            lower_ci = summary['exp(coef) lower 95%']
            upper_ci = summary['exp(coef) upper 95%']
            p_val = summary['p']
            
            results.append({
                "Gene": gene,
                "Hazard_Ratio": hr,
                "CI_Lower": lower_ci,
                "CI_Upper": upper_ci,
                "P_value": p_val,
                "Role": "Risk" if hr > 1 else "Protective"
            })
        except Exception: pass

        # Plot KM (Individual Gene)
        plot_km_with_stats(
            df_analysis[gene].values,
            gene,
            df_analysis['T'].values,
            df_analysis['E'].values,
            out_path / f"KM_{gene}.png"
        )
    
    # --- Biosignature (Combined) ---
    local_cph = None
    local_risk_median = None

    if len(available_feats) > 1:
        print(f"   >> Computing Biosignature for {len(available_feats)} features...")
        
        # Strategy 3: Refit on current dataset (Local Model, Median Threshold)
        # This is strictly local evaluation (Re-discovery)
        try:
            local_cph = CoxPHFitter(penalizer=0.1) # Add slight penalizer
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                local_cph.fit(df_analysis, duration_col='T', event_col='E')
            
            risk_scores_refit = local_cph.predict_partial_hazard(df_analysis)
            local_risk_median = np.median(risk_scores_refit)
            
            plot_km_with_stats(
                risk_scores_refit.values,
                "Biosignature_Refit",
                df_analysis['T'].values,
                df_analysis['E'].values,
                out_path / "KM_Biosignature_Refit.png",
                add_to_results=True
            )
        except Exception as e:
            print(f"   >> [WARN] Could not plot Biosignature (Refit): {e}")

        # Strategies 1 & 2: Transfer Learning (if external model provided)
        if trained_model is not None:
            try:
                print(f"   >> [INFO] Computing Biosignature Transfer Strategies...") 
                # Predict Risk Score using External Model
                # Need to match feature columns passed to model
                risk_scores_ext = trained_model.predict_partial_hazard(df_analysis)
                
                # Strategy 1: Fixed Threshold (from Selection)
                # "Validation of Cutoff"
                if trained_median is not None:
                     print(f"      >> Strategy 1: Transfer Fixed Threshold (Thr={trained_median:.4f})")
                     plot_km_with_stats(
                        risk_scores_ext.values,
                        "Biosignature_Transfer_FixedThr",
                        df_analysis['T'].values,
                        df_analysis['E'].values,
                        out_path / "KM_Biosignature_Transfer_FixedThr.png",
                        add_to_results=True,
                        fixed_threshold=trained_median
                    )

                # Strategy 2: Adaptive Threshold (Median of Verify)
                # "Validation of Score"
                print(f"      >> Strategy 2: Transfer Adaptive Median Threshold")
                plot_km_with_stats(
                    risk_scores_ext.values,
                    "Biosignature_Transfer_MedianThr",
                    df_analysis['T'].values,
                    df_analysis['E'].values,
                    out_path / "KM_Biosignature_Transfer_MedianThr.png",
                    add_to_results=True
                )
            except Exception as e:
                print(f"   >> [WARN] Could not plot Biosignature (Transfer): {e}")

    # --- 3. Lưu bảng tổng hợp ---
    if results:
        res_df = pd.DataFrame(results)
        # Sort so Biosignature (if any) is easy to find, often P-value based
        res_df = res_df.sort_values("P_value")
        save_path = Path(outdir) / csv_name
        res_df.to_csv(save_path, index=False)
        print(f"[DONE] Đã lưu phân tích đơn biến{label_str} tại: {save_path}")
        print(res_df.to_string())
    
    return local_cph, local_risk_median

# -------------------------------
# Main Pipeline
# -------------------------------

def build_stabl_cox(n_bootstraps=500, random_state=42, debug_dir=None, model_type="coxnet"):
    if model_type == "coxnet":
        lambda_grid = "auto"
        base = CoxnetSurvivalAnalysis(l1_ratio=0.9, fit_baseline_model=True, alpha_min_ratio=0.01, max_iter=100000, tol=1e-7)
    elif model_type == "gradient_boosting":
        base = ComponentwiseGradientBoostingSurvivalAnalysis(loss="coxph", random_state=random_state)
        lambda_grid = {"n_estimators": np.arange(10, 210, 10)} 
    elif model_type == "rsf":
        base = RandomSurvivalForest(n_estimators=100, min_samples_split=10, max_depth=10, max_features="sqrt", n_jobs=-1, random_state=random_state)
        lambda_grid = {"min_samples_leaf": np.arange(2, 22, 2)}
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


    stabl_cox = Stabl(
        base_estimator=base,
        lambda_grid=lambda_grid,
        n_bootstraps=n_bootstraps,
        artificial_type="knockoff",
        artificial_proportion=1.0,
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


def run_pipeline(args):
    os.makedirs(args.outdir, exist_ok=True)
    selected_features = []

    # ---------------------------------------------------------
    # 1. Load Selection Data (Always load for analysis purpose)
    # ---------------------------------------------------------
    print("\n[STEP 1] Loading Selection Data...")
    X_sel = load_counts(args.selection_counts, num_genes=args.num_genes, debug_dir=args.debug_dir)
    y_sel = load_clinical(args.selection_clinical, target_type=args.target_type)
    X_sel, y_sel = align_X_y(X_sel, y_sel)
    
    # Preprocessing
    print(f"[INFO] Preprocessing Selection Data (LIF, Impute, Scale)...")
    
    # Setup log dir for removed features
    removed_dir = Path(args.outdir) / "removed_features_logs"
    removed_dir.mkdir(parents=True, exist_ok=True)

    # 1. LowInfoFilter - DISABLED
    # lif = LowInfoFilter(max_nan_fraction=0.2)
    # lif.fit(X_sel)
    
    # # Identify dropped by LIF
    # cols_kept_lif = lif.get_feature_names_out() if hasattr(lif, "get_feature_names_out") else X_sel.columns[lif.get_support()]
    # dropped_lif = list(set(X_sel.columns) - set(cols_kept_lif))
    # if dropped_lif:
    #     print(f"[INFO] LowInfoFilter dropped {len(dropped_lif)} features.")
    #     pd.Series(dropped_lif, name="Dropped_LIF").to_csv(removed_dir / "dropped_LIF.csv", index=False)

    # X_sel_np = lif.transform(X_sel)
    # X_sel = pd.DataFrame(X_sel_np, index=X_sel.index, columns=cols_kept_lif)
    pass
    
    imputer = SimpleImputer(strategy="median")
    X_sel = pd.DataFrame(imputer.fit_transform(X_sel), index=X_sel.index, columns=X_sel.columns)
    scaler = StandardScaler()
    X_sel = pd.DataFrame(scaler.fit_transform(X_sel), index=X_sel.index, columns=X_sel.columns)
    
    # Save scaler features for later use on Verify data
    scaler_feature_names = X_sel.columns.tolist()

    if args.selected_features_file:
        print(f"\n[INFO] Skipping Selection. Loading features from {args.selected_features_file}...")
        try:
            sf_df = pd.read_csv(args.selected_features_file)
            if "Selected_Features" in sf_df.columns:
                selected_features = sf_df["Selected_Features"].tolist()
            else:
                selected_features = sf_df.iloc[:, 0].tolist()
            print(f"[INFO] Loaded {len(selected_features)} features from file.")
            
            # Save copy to current output dir
            pd.Series(selected_features, name="Selected_Features").to_csv(Path(args.outdir) / "selected_features.csv", index=False)
            
        except Exception as e:
            print(f"[ERROR] Failed to load selected features from file: {e}")
            return
    else:
        if args.univariate_cox:
            genes_pre_cox = set(X_sel.columns)
            X_sel = run_univariate_cox(X_sel, y_sel, p_thresh=args.univariate_p, debug_dir=args.debug_dir)
            genes_post_cox = set(X_sel.columns)
            dropped_cox = list(genes_pre_cox - genes_post_cox)
            if dropped_cox:
                print(f"[INFO] Univariate Cox dropped {len(dropped_cox)} features.")
                removed_dir = Path(args.outdir) / "removed_features_logs"
                removed_dir.mkdir(parents=True, exist_ok=True)
                pd.Series(dropped_cox, name="Dropped_UnivariateCox").to_csv(removed_dir / "dropped_univariate_cox.csv", index=False)

        # ---------------------------------------------------------
        # 2. Run STABL Feature Selection
        # ---------------------------------------------------------
        print("\n[STEP 2] Running STABL Feature Selection...")
        print(f"[INFO] Number of genes input to STABL: {X_sel.shape[1]} (Samples: {X_sel.shape[0]})")

        stabl_cox = build_stabl_cox(n_bootstraps=args.n_boot, random_state=args.seed, debug_dir=args.debug_dir, model_type=args.model_type)
        stabl_cox.fit(X_sel, y_sel)
        
        selected_mask = stabl_cox.get_support()
        if hasattr(stabl_cox, "feature_names_in_"):
            selected_features = stabl_cox.feature_names_in_[selected_mask]
        else:
            selected_features = X_sel.columns[selected_mask]
            
        print(f"[INFO] STABL selected {len(selected_features)} features: {list(selected_features)}")
        pd.Series(selected_features, name="Selected_Features").to_csv(Path(args.outdir) / "selected_features.csv", index=False)
        try:
            save_stabl_results(stabl_cox, Path(args.outdir), X_sel, y_sel, task_type="survival")
        except Exception: pass

    if len(selected_features) == 0:
        print("[ERROR] No features selected. Cannot proceed to training.")
        return

    # --- NEW: Analyze on Selection Dataset ---
    print("\n[ANALYSIS] Analyzing Individual Genes on Selection Dataset...")
    sel_model, sel_median = analyze_individual_genes(X_sel, y_sel, selected_features, args.outdir, dataset_label="selection_dataset")

    # ---------------------------------------------------------
    # 3. Load Verify Data
    # ---------------------------------------------------------
    print("\n[STEP 3] Loading Verify Data...")
    # NOTE: num_genes is ONLY for selection dataset cutting. Verify dataset should be loaded fully
    # to find matching genes later.
    X_verify = load_counts(args.verify_counts, num_genes=None, debug_dir=args.debug_dir) 
    y_verify = load_clinical(args.verify_clinical, target_type=args.target_type)
    X_verify, y_verify = align_X_y(X_verify, y_verify)

    # --- FILTER SELECTED FEATURES FOUND IN VERIFY ---
    verified_genes = [f for f in selected_features if f in X_verify.columns]
    print(f"\n[INFO] {len(verified_genes)}/{len(selected_features)} selected features found in Verify dataset.")
    pd.Series(verified_genes, name="Verified_Features").to_csv(Path(args.outdir) / "verified_features.csv", index=False)
    
    if len(selected_features) > len(verified_genes):
        missing = list(set(selected_features) - set(verified_genes))
        print(f"       Missing features (skipped): {missing}")
    
    if len(verified_genes) == 0:
        print("[ERROR] No selected features found in Verify dataset. Exiting verification steps.")
        return

    # 3a. Analyze on Entire Verify Dataset
    print("\n[ANALYSIS] Analyzing Individual Genes on Entire Verify Dataset (Verified Genes only)...")
    
    # Base Imputed Data
    X_verify_base = X_verify.copy()
    X_verify_base = X_verify_base[verified_genes].replace([np.inf, -np.inf], np.nan)
    X_verify_base = X_verify_base.fillna(X_verify_base.median())

    # --- PREPARE TRANSFER MODEL (TRAIN ON SELECTION WITH VERIFIED GENES ONLY) ---
    transfer_model = None
    transfer_median = None
    
    if len(verified_genes) > 0 and 'CoxPHFitter' in globals():
        try:
            print("\n[INFO] Retraining Transfer Model on Selection Data using ONLY Verified Genes...")
            X_sel_sub = X_sel[verified_genes].copy()
            df_sel_train = X_sel_sub.copy()
            df_sel_train['T'] = y_sel['time'].values
            df_sel_train['E'] = y_sel['event'].astype(int).values
            
            transfer_model = CoxPHFitter(penalizer=0.1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                transfer_model.fit(df_sel_train, duration_col='T', event_col='E')
            
            risk_scores_sel = transfer_model.predict_partial_hazard(df_sel_train)
            transfer_median = np.median(risk_scores_sel)
            print(f"       >> Transfer Model retrained successfully on {len(verified_genes)} genes. Median Risk (Sel) = {transfer_median:.4f}")
        except Exception as e:
            print(f"       >> [WARN] Could not retrain transfer model: {e}")
            transfer_model = None
            transfer_median = None

    # --- VARIANT 1: Selection Scaling (No Refit) ---
    print("\n[INFO] [Variant 1] Scaling Verify Data using Selection Data statistics (No Refit)...")
    X_verify_sel = X_verify_base.copy()
    
    if 'scaler_feature_names' in locals():
        scaler_feats_set = set(scaler_feature_names) 
        valid_cols = [col for col in X_verify_sel.columns if col in scaler_feats_set]
        missing_cols = list(set(X_verify_sel.columns) - set(valid_cols))
        
        if missing_cols:
            print(f"[WARN] {len(missing_cols)} features not found in Selection Scaler: {missing_cols}")
            X_verify_sel = X_verify_sel[valid_cols]

        if not X_verify_sel.empty:
            feat_map = {name: i for i, name in enumerate(scaler_feature_names)}
            indices = [feat_map[col] for col in X_verify_sel.columns]
            mean_vals = scaler.mean_[indices]
            scale_vals = scaler.scale_[indices]
            X_verify_sel = (X_verify_sel - mean_vals) / scale_vals
    
    analyze_individual_genes(
        X_verify_sel, y_verify, verified_genes, args.outdir, 
        dataset_label="verify_selection_scaling",
        trained_model=transfer_model, 
        trained_median=transfer_median 
    )

    # --- VARIANT 2: Independent Scaling (Refit on Verify) ---
    print("\n[INFO] [Variant 2] Independent Scaling (Fit on Verify)...")
    scaler_ver = StandardScaler()
    X_verify_ind = pd.DataFrame(
        scaler_ver.fit_transform(X_verify_base), 
        index=X_verify_base.index, 
        columns=X_verify_base.columns
    )
    analyze_individual_genes(
        X_verify_ind, y_verify, verified_genes, args.outdir, 
        dataset_label="verify_independent_scaling",
        trained_model=transfer_model,
        trained_median=transfer_median 
    )

    # --- VARIANT 3: No Scaling ---
    print("\n[INFO] [Variant 3] No Scaling (Imputed only)...")
    analyze_individual_genes(
        X_verify_base, y_verify, verified_genes, args.outdir, 
        dataset_label="verify_no_scaling",
        trained_model=transfer_model, 
        trained_median=transfer_median 
    )

    # 3b. Split for Final Models
    print(f"\n[STEP 3b] Splitting Verify Data (Test Ratio={args.verify_split_ratio})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_verify, y_verify, test_size=args.verify_split_ratio, random_state=args.seed, stratify=y_verify['event']
    )

    # ---------------------------------------------------------
    # 5. Preprocessing for Final Models
    # ---------------------------------------------------------
    print("\n[STEP 5] Preprocessing and Training Final Models...")
    
    # Helper to preprocess
    def preprocess_data(X_tr, X_te, features=None):
        if features is not None:
            # Case 1: Specific features requested (Verification of Selected Genes)
            # STRICTLY keep these genes found in Verify. NO LowInfoFilter.
            print("[INFO] Preprocessing with Verified Genes: Skipping LowInfoFilter to preserve selected set.")
            valid_feats = [f for f in features if f in X_tr.columns]
            if len(valid_feats) < len(features):
                print(f"[WARN] Preprocessing: requested {len(features)} features, found {len(valid_feats)}.")
            
            X_tr = X_tr[valid_feats]
            X_te = X_te[valid_feats]
            # NO LIF applied here
            
        else:
            # Case 2: Full Features (Baseline)
            # Apply LowInfoFilter to remove constant/high-nan features
            print("[INFO] Preprocessing Full Features: Skipping LowInfoFilter (DISABLED).")
            # lif = LowInfoFilter(max_nan_fraction=0.2)
            # X_tr_np = lif.fit_transform(X_tr)
            # cols = lif.get_feature_names_out() if hasattr(lif, "get_feature_names_out") else X_tr.columns[lif.get_support()]
            # X_te_np = lif.transform(X_te)
            
            # X_tr = pd.DataFrame(X_tr_np, index=X_tr.index, columns=cols)
            # X_te = pd.DataFrame(X_te_np, index=X_te.index, columns=cols)
            pass
        
        imputer = SimpleImputer(strategy="median")
        X_tr = pd.DataFrame(imputer.fit_transform(X_tr), index=X_tr.index, columns=X_tr.columns)
        X_te = pd.DataFrame(imputer.transform(X_te), index=X_te.index, columns=X_te.columns)
        
        scaler = StandardScaler()
        X_tr = pd.DataFrame(scaler.fit_transform(X_tr), index=X_tr.index, columns=X_tr.columns)
        X_te = pd.DataFrame(scaler.transform(X_te), index=X_te.index, columns=X_te.columns)
        return X_tr, X_te

    # 1. Selected Features Data
    X_train_sub, X_test_sub = preprocess_data(X_train.copy(), X_test.copy(), features=verified_genes)
    # 2. Full Features Data
    print("[INFO] Processing Full Features Data (this might take a while)...")
    X_train_full, X_test_full = preprocess_data(X_train.copy(), X_test.copy())

    # Structured arrays for sksurv
    y_train_surv = Surv.from_arrays(event=y_train['event'].astype(bool).values, time=y_train['time'].values)
    y_test_surv = Surv.from_arrays(event=y_test['event'].astype(bool).values, time=y_test['time'].values)

    def get_models(seed):
        return {
            "CoxPH": CoxPHSurvivalAnalysis(),
            "Coxnet": CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01, fit_baseline_model=True),
            "GradientBoosting": ComponentwiseGradientBoostingSurvivalAnalysis(loss="coxph", random_state=seed),
            "RSF": RandomSurvivalForest(n_estimators=100, min_samples_split=10, min_samples_leaf=15, max_features="sqrt", n_jobs=-1, random_state=seed)
        }

    results_list = []

    # --- 1. Evaluate on SELECTED Features ---
    print(f"\n--- Evaluating Models on SELECTED Features ({X_train_sub.shape[1]} features) ---")
    models_sub = get_models(args.seed)
    for name, model in models_sub.items():
        if name == "CoxPH" and X_train_sub.shape[1] > X_train_sub.shape[0]:
            print(f"Skipping {name} (p > n)")
            continue
            
        try:
            model.fit(X_train_sub, y_train_surv)
            
            # --- GỌI HÀM ĐÁNH GIÁ NÂNG CAO ---
            eval_res = evaluate_survival_model_comprehensive(
                model, X_train_sub, y_train_surv, X_test_sub, y_test_surv, 
                model_name=f"{name}_Selected", outdir=args.outdir
            )
            
            res_entry = {
                "Feature_Set": "Selected",
                "Model": name,
                "Num_Features": X_train_sub.shape[1]
            }
            res_entry.update(eval_res)
            results_list.append(res_entry)

        except Exception as e:
            print(f"[{name}] Failed on Selected Features: {e}")
            traceback.print_exc()

    # --- 2. Evaluate on FULL Features ---
    print(f"\n--- Evaluating Models on FULL Features ({X_train_full.shape[1]} features) ---")
    models_full = get_models(args.seed)
    for name, model in models_full.items():
        if name == "CoxPH" and X_train_full.shape[1] > X_train_full.shape[0]:
            print(f"Skipping {name} on Full Features (p > n)")
            continue

        try:
            print(f"Training {name} on Full Features...")
            model.fit(X_train_full, y_train_surv)
            
            eval_res = evaluate_survival_model_comprehensive(
                model, X_train_full, y_train_surv, X_test_full, y_test_surv, 
                model_name=f"{name}_Full", outdir=args.outdir
            )
            
            res_entry = {
                "Feature_Set": "Full",
                "Model": name,
                "Num_Features": X_train_full.shape[1]
            }
            res_entry.update(eval_res)
            results_list.append(res_entry)

        except Exception as e:
            print(f"[{name}] Failed on Full Features: {e}")

    # ---------------------------------------------------------
    # 6. Save Final Results
    # ---------------------------------------------------------
    print("\n[STEP 6] Saving Final Comparison...")
    res_df = pd.DataFrame(results_list)
    res_df.to_csv(Path(args.outdir) / "final_model_comparison.csv", index=False)
    print(res_df)
    print(f"\n[DONE] Results saved to {args.outdir}")


def main(args):
    log_buffer = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

        # Save run parameters
        import json
        try:
            with open(Path(args.debug_dir) / "run_params.json", "w") as f:
                json.dump(vars(args), f, indent=4)
        except Exception as e:
            print(f"[WARN] Could not save run params: {e}")

        log_buffer = io.StringIO()
        class Tee(object):
            def __init__(self, stream, buffer):
                self.stream = stream
                self.buffer = buffer
            def write(self, message):
                self.stream.write(message)
                self.buffer.write(message)
            def flush(self):
                self.stream.flush()
            def fileno(self): return self.stream.fileno()
        sys.stdout = Tee(original_stdout, log_buffer)
        sys.stderr = Tee(original_stderr, log_buffer)

    try:
        run_pipeline(args)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
    finally:
        if log_buffer:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            try:
                with open(Path(args.debug_dir) / "run_log.txt", "w", encoding="utf-8") as f:
                    f.write(log_buffer.getvalue())
            except Exception: pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run STABL Cox: Selection on one dataset, Verify on another (split internally) - FIXED VERSION")
    
    # Selection Data
    parser.add_argument("--selection_counts", required=True, help="Path to counts file for Feature Selection")
    parser.add_argument("--selection_clinical", required=True, help="Path to clinical file for Feature Selection")
    
    # Verify Data (To be split)
    parser.add_argument("--verify_counts", required=True, help="Path to counts file for Verification")
    parser.add_argument("--verify_clinical", required=True, help="Path to clinical file for Verification")
    parser.add_argument("--verify_split_ratio", type=float, default=0.3, help="Ratio of Verify data to use for Testing (default: 0.3)")
    
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--n_boot", type=int, default=200, help="Number of bootstraps for STABL")
    parser.add_argument("--seed", type=int, default=42, help="random_state")
    parser.add_argument("--num_genes", type=int, default=None, help="Keep only the first N genes in file order (for Selection data).")
    parser.add_argument("--debug_dir", type=str, default=None, help="Directory to save debug info")
    parser.add_argument("--model_type", type=str, default="coxnet", choices=["coxnet", "gradient_boosting", "rsf"], help="Base estimator type")
    parser.add_argument("--univariate_cox", action="store_true", help="Run univariate Cox selection before Stabl")
    parser.add_argument("--univariate_p", type=float, default=0.05, help="P-value threshold for univariate Cox")
    parser.add_argument("--selected_features_file", type=str, default=None, help="Path to file containing selected features (skip selection step)")
    parser.add_argument("--target_type", type=str, default="OS", choices=["OS", "RFS"], help="Survival target: OS or RFS [Default: OS]")

    args = parser.parse_args()
    main(args)