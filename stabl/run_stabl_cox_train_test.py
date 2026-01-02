# import os
# import argparse
# import numpy as np
# import pandas as pd
# from pathlib import Path
# import sys
# import io
# import traceback
# import warnings

# # Local imports
# # Assuming this script is run as a module: python -m stabl.run_stabl_cox_train_test
# # or from the root: python stabl/run_stabl_cox_train_test.py (if stabl is in pythonpath)
# # We will try relative imports if package, else absolute.
# try:
#     from .stabl import Stabl, save_stabl_results
# except ImportError:
#     # If running as script from stabl/ folder
#     sys.path.append(str(Path(__file__).parent.parent))
#     from stabl.stabl import Stabl, save_stabl_results

# from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
# from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
# from sksurv.metrics import concordance_index_censored
# from sksurv.util import Surv
# from joblib import Parallel, delayed
# from sklearn.model_selection import train_test_split

# # -------------------------------
# # Utils (Copied from run_stabl_cox_FIXED.py)
# # -------------------------------

# def detect_sep(path):
#     """Try to guess separator: prefer tab if '\t' found in first 1KB."""
#     with open(path, "r", encoding="utf-8", errors="ignore") as f:
#         head = f.read(1024)
#     if "\t" in head and "," not in head:
#         return "\t"
#     if "\t" in head and "," in head:
#         return "\t"
#     return ","


# def load_counts(counts_path, num_genes: int | None = None, debug_dir: str | None = None):
#     """
#     Read counts: first 2 columns = gene_name, entrez_id; remaining columns = samples.
#     Returns:
#       X (DataFrame): samples x genes (transposed, columns = gene_symbol)
#     If num_genes is set, keep only the first num_genes genes in the *file order*.
#     """
#     sep = detect_sep(counts_path)
#     df = pd.read_csv(counts_path, sep=sep, header=0)
#     if df.shape[1] < 4:
#         raise ValueError("File counts must have >= 4 columns (gene, entrez, and >=2 samples).")

#     gene_col = df.columns[0]
#     sample_cols = df.columns[2:]

#     # Build deterministic gene order
#     genes_in_order = pd.Index(df[gene_col].astype(str).tolist())
#     first_occ_mask = ~genes_in_order.duplicated()
#     unique_genes_in_order = genes_in_order[first_occ_mask]

#     # Matrix genes x samples
#     expr = df[sample_cols].copy()
#     expr.index = df[gene_col].astype(str).values
    
#     # Aggregate duplicates by median WITHOUT sorting
#     expr = expr.groupby(expr.index, sort=False).median()

#     # Reindex to first-appearance order
#     expr = expr.reindex(unique_genes_in_order.intersection(expr.index))

#     # Cap to first N genes if requested
#     if num_genes is not None:
#         expr = expr.iloc[:num_genes, :]

#     # Transpose: samples x genes, coerce numeric
#     X = expr.T.apply(pd.to_numeric, errors="coerce")
#     return X


# def load_clinical(clinical_path):
#     """
#     Read clinical with columns: sample id, OS, censored.
#     Convention: censored = 0 (alive), 1 (dead) -> event=False/True.
#     Returns DataFrame indexed by sample_id, columns ['time','event'] (event bool).
#     """
#     sep = detect_sep(clinical_path)
#     clin = pd.read_csv(clinical_path, sep=sep, header=0)
#     clin.columns = [c.replace("\ufeff", "") for c in clin.columns]

#     # Standardize column names
#     ren = {c: c.strip().lower() for c in clin.columns}
#     clin = clin.rename(columns=ren)

#     # Map column names
#     col_map = {}
#     # sample id
#     for cand in ["sample id", "sample_id", "sample", "id", "case_id", "patient_id", "case submitter id", "cgga_id"]:
#         if cand in clin.columns:
#             col_map["sample_id"] = cand
#             break
#     # OS/time
#     for cand in ["os", "time", "overall_survival", "overall survival", "survival_time", "survival time", "days_to_death", "days to death"]:
#         if cand in clin.columns:
#             col_map["time"] = cand
#             break
#     # censored/status/event
#     for cand in ["censored", "censor", "status", "event", "vital_status", "vital status"]:
#         if cand in clin.columns:
#             col_map["censored"] = cand
#             break

#     if set(col_map.keys()) != {"sample_id", "time", "censored"}:
#         raise ValueError(
#             f"Cannot find all required columns (sample id, os/time, censored). "
#             f"Mapped: {col_map}. Available: {list(clin.columns)}"
#         )

#     # Extract raw data
#     sid_raw = clin[col_map["sample_id"]].astype(str)
#     time_raw = clin[col_map["time"]]
#     cens_raw = clin[col_map["censored"]]

#     # Normalize time -> numeric
#     time_str = (time_raw.astype(str)
#                 .str.replace(r"[^0-9\.,\-]", "", regex=True)
#                 .str.replace(",", ".", regex=False)
#                )
#     time = pd.to_numeric(time_str, errors="coerce")

#     # Normalize censored: allow text variants
#     cens_str = cens_raw.astype(str).str.strip().str.lower()
#     text_map = {
#         "alive": "0", "censored": "0", "living": "0", "no_event": "0", "no event": "0",
#         "dead": "1", "deceased": "1", "event": "1", "died": "1",
#         "0.0": "0", "1.0": "1"
#     }
#     cens_norm = cens_str.map(lambda x: text_map.get(x, x))
#     cens = pd.to_numeric(cens_norm, errors="coerce")

#     # 0 = alive -> False, 1 = dead -> True
#     event = (cens == 1).astype("boolean")

#     # Use .values to avoid index alignment
#     sid_idx = pd.Index(sid_raw.astype(str).to_numpy(), name="sample_id")
#     y = pd.DataFrame(
#         {
#             "time":  time.to_numpy(),
#             "event": event.to_numpy(),
#         },
#         index=sid_idx
#     )

#     # Handle duplicate sample_ids
#     if y.index.duplicated().any():
#         y = y[~y.index.duplicated(keep="first")]

#     y = y.dropna(subset=["time", "event"])
#     return y


# def align_X_y(X, y):
#     """
#     Intersect samples between X and y; sort index for consistency.
#     """
#     inter = X.index.astype(str).intersection(y.index.astype(str))
#     X2 = X.loc[inter].sort_index()
#     y2 = y.loc[inter].sort_index()
#     # Drop samples with NaN time/event
#     mask = (~y2['time'].isna()) & (~y2['event'].isna())
#     X2 = X2.loc[mask]
#     y2 = y2.loc[mask]
#     return X2, y2


# def run_univariate_cox(X, y, p_thresh=0.05, debug_dir=None):
#     """
#     Run univariate Cox PH regression for each feature in X against y.
#     Select features with p-value < p_thresh.
#     """
#     print(f"[INFO] Running univariate Cox selection (p < {p_thresh})...")
#     try:
#         from lifelines import CoxPHFitter
#     except ImportError:
#         print("[WARNING] lifelines not installed. Skipping univariate Cox selection.")
#         return X

#     genes = X.columns
#     n_genes = len(genes)
    
#     # Prepare data arrays once
#     times = y['time'].values
#     events = y['event'].values.astype(int) # lifelines prefers 0/1
    
#     def fit_one_gene(gene, values):
#         try:
#             # Create mini dataframe
#             df = pd.DataFrame({
#                 'feature': values,
#                 'T': times,
#                 'E': events
#             })
#             # Drop NaNs if any (though align_X_y should have handled it)
#             df = df.dropna()
            
#             if df['feature'].nunique() < 2:
#                 return (gene, 1.0) # Constant feature
                
#             cph = CoxPHFitter()
#             # suppress warnings
#             with warnings.catch_warnings():
#                 warnings.simplefilter("ignore")
#                 cph.fit(df, duration_col='T', event_col='E')
            
#             # Get p-value of the 'feature' coefficient
#             pval = cph.summary.loc['feature', 'p']
#             return (gene, pval)
#         except Exception:
#             return (gene, 1.0)

#     # Run in parallel
#     print(f"[INFO] Fitting {n_genes} univariate models...")
#     results = Parallel(n_jobs=-1, verbose=5)(
#         delayed(fit_one_gene)(g, X[g].values) 
#         for g in genes
#     )
    
#     res_df = pd.DataFrame(results, columns=['gene', 'p_value'])
    
#     # Save results
#     if debug_dir:
#         os.makedirs(debug_dir, exist_ok=True)
#         res_df.to_csv(Path(debug_dir) / "univariate_cox_results.csv", index=False)
        
#     # Filter
#     selected = res_df[res_df['p_value'] < p_thresh]['gene'].tolist()
#     print(f"[INFO] Univariate Cox: {len(selected)}/{n_genes} features passed p < {p_thresh}")
    
#     if len(selected) == 0:
#         print("[WARNING] No features passed univariate filter! Reverting to all features.")
#         return X
        
#     return X[selected]


# def build_stabl_cox(n_bootstraps=500, random_state=42, debug_dir=None, model_type="coxnet"):
#     """
#     Build STABL Cox estimator.
#     """
#     if model_type == "coxnet":
#         lambda_grid = "auto"
#         base = CoxnetSurvivalAnalysis(
#             l1_ratio=0.9,
#             fit_baseline_model=True,
#             alpha_min_ratio=0.01,
#             max_iter=1000000,
#             tol=1e-7,
#             verbose=False
#         )
#     elif model_type == "gradient_boosting":
#         base = ComponentwiseGradientBoostingSurvivalAnalysis(
#             loss="coxph",
#             random_state=random_state,
#             verbose=0
#         )
#         lambda_grid = {"n_estimators": np.arange(10, 200, 20)} 
#     else:
#         raise ValueError(f"Unknown model_type: {model_type}")

#     stabl_cox = Stabl(
#         base_estimator=base,
#         lambda_grid=lambda_grid,
#         n_bootstraps=n_bootstraps,
#         artificial_type="knockoff",
#         artificial_proportion=1,
#         sample_fraction=0.5,
#         replace=False,
#         bootstrap_threshold="median",
#         fdr_threshold_range=np.arange(0.05, 1.01, 0.05),
#         explore=True,
#         n_explore=5,
#         task_type="survival",
#         random_state=random_state,
#         n_jobs=-1,
#         verbose=1,
#         debug_dir=debug_dir
#     )
#     return stabl_cox

# def run_pipeline(args):
#     os.makedirs(args.outdir, exist_ok=True)
    
#     # ---------------------------------------------------------
#     # 1. Load Selection Data
#     # ---------------------------------------------------------
#     print("\n[STEP 1] Loading Selection Data...")
#     X_sel = load_counts(args.selection_counts, num_genes=args.num_genes, debug_dir=args.debug_dir)
#     y_sel = load_clinical(args.selection_clinical)
#     X_sel, y_sel = align_X_y(X_sel, y_sel)
#     print(f"Selection Data: {X_sel.shape[0]} samples, {X_sel.shape[1]} genes")

#     # Univariate Cox Filter (Optional)
#     if args.univariate_cox:
#         X_sel = run_univariate_cox(X_sel, y_sel, p_thresh=args.univariate_p, debug_dir=args.debug_dir)

#     # ---------------------------------------------------------
#     # 2. Run STABL Feature Selection
#     # ---------------------------------------------------------
#     print("\n[STEP 2] Running STABL Feature Selection...")
#     stabl_cox = build_stabl_cox(
#         n_bootstraps=args.n_boot, 
#         random_state=args.seed, 
#         debug_dir=args.debug_dir,
#         model_type=args.model_type
#     )
    
#     stabl_cox.fit(X_sel, y_sel)
    
#     # Get selected features
#     selected_mask = stabl_cox.get_support()
#     if hasattr(stabl_cox, "feature_names_in_"):
#         selected_features = stabl_cox.feature_names_in_[selected_mask]
#     else:
#         selected_features = X_sel.columns[selected_mask]
        
#     print(f"[INFO] STABL selected {len(selected_features)} features: {list(selected_features)}")
    
#     # Save selected features
#     pd.Series(selected_features, name="Selected_Features").to_csv(
#         Path(args.outdir) / "selected_features.csv", index=False
#     )
    
#     # Save STABL results (plots etc)
#     try:
#         save_stabl_results(
#             stabl_cox, 
#             Path(args.outdir), 
#             X_sel, 
#             y_sel, 
#             task_type="survival"
#         )
#     except Exception as e:
#         print(f"[WARNING] Failed to save STABL plots: {e}")

#     if len(selected_features) == 0:
#         print("[ERROR] No features selected. Cannot proceed to training.")
#         return

#     # ---------------------------------------------------------
#     # 3. Load Verify Data and Split
#     # ---------------------------------------------------------
#     print("\n[STEP 3] Loading Verify Data and Splitting...")
    
#     # Load Verify Data
#     X_verify = load_counts(args.verify_counts, num_genes=None) # Load all genes first
#     y_verify = load_clinical(args.verify_clinical)
#     X_verify, y_verify = align_X_y(X_verify, y_verify)
    
#     print(f"Verify Data (Total): {X_verify.shape[0]} samples")

#     # Split Verify Data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_verify, y_verify, 
#         test_size=args.verify_split_ratio, 
#         random_state=args.seed,
#         stratify=y_verify['event'] # Stratify by event status
#     )
    
#     print(f"Verify Train Split: {X_train.shape[0]} samples")
#     print(f"Verify Test Split: {X_test.shape[0]} samples")

#     # ---------------------------------------------------------
#     # 4. Subset to Selected Features
#     # ---------------------------------------------------------
#     # Ensure all selected features exist in Verify Data
#     missing_verify = set(selected_features) - set(X_verify.columns)
    
#     if missing_verify:
#         print(f"[WARNING] {len(missing_verify)} selected features missing in Verify Data. Filling with 0.")
#         for f in missing_verify:
#             X_train[f] = 0.0
#             X_test[f] = 0.0
            
#     X_train_sub = X_train[selected_features]
#     X_test_sub = X_test[selected_features]

#     # ---------------------------------------------------------
#     # 5. Train Final Cox Model
#     # ---------------------------------------------------------
#     print("\n[STEP 4] Training Final Cox Model on Verify Train Split...")
    
#     # Prepare structured arrays for sksurv
#     y_train_surv = Surv.from_arrays(
#         event=y_train['event'].astype(bool).values, 
#         time=y_train['time'].values
#     )
#     y_test_surv = Surv.from_arrays(
#         event=y_test['event'].astype(bool).values, 
#         time=y_test['time'].values
#     )
    
#     # Use standard CoxPH
#     final_model = CoxPHSurvivalAnalysis()
#     final_model.fit(X_train_sub, y_train_surv)
    
#     # ---------------------------------------------------------
#     # 6. Evaluate
#     # ---------------------------------------------------------
#     print("\n[STEP 5] Evaluating on Verify Test Split...")
    
#     # C-index on Train
#     train_c = final_model.score(X_train_sub, y_train_surv)
#     print(f"Verify Train C-index: {train_c:.4f}")
    
#     # C-index on Test
#     test_c = final_model.score(X_test_sub, y_test_surv)
#     print(f"Verify Test C-index: {test_c:.4f}")
    
#     # Save results
#     with open(Path(args.outdir) / "final_metrics.txt", "w") as f:
#         f.write(f"Verify Train C-index: {train_c:.4f}\n")
#         f.write(f"Verify Test C-index: {test_c:.4f}\n")
#         f.write(f"Selected Features: {len(selected_features)}\n")
#         f.write(f"Verify Total Samples: {X_verify.shape[0]}\n")
#         f.write(f"Verify Train Samples: {X_train.shape[0]}\n")
#         f.write(f"Verify Test Samples: {X_test.shape[0]}\n")

#     print(f"\n[DONE] Results saved to {args.outdir}")


# def main():
#     parser = argparse.ArgumentParser(description="Run STABL Cox: Selection on one dataset, Verify on another (split internally).")
    
#     # Selection Data
#     parser.add_argument("--selection_counts", required=True, help="Path to counts file for Feature Selection")
#     parser.add_argument("--selection_clinical", required=True, help="Path to clinical file for Feature Selection")
    
#     # Verify Data (To be split)
#     parser.add_argument("--verify_counts", required=True, help="Path to counts file for Verification")
#     parser.add_argument("--verify_clinical", required=True, help="Path to clinical file for Verification")
#     parser.add_argument("--verify_split_ratio", type=float, default=0.3, help="Ratio of Verify data to use for Testing (default: 0.3)")
    
#     parser.add_argument("--outdir", required=True, help="Output directory")
#     parser.add_argument("--n_boot", type=int, default=200, help="Number of bootstraps for STABL")
#     parser.add_argument("--seed", type=int, default=42, help="random_state")
#     parser.add_argument(
#         "--num_genes",
#         type=int,
#         default=None,
#         help="Keep only the first N genes in file order (for Selection data)."
#     )
#     parser.add_argument("--debug_dir", type=str, default=None, help="Directory to save debug info")
#     parser.add_argument(
#         "--model_type", 
#         type=str, 
#         default="coxnet", 
#         choices=["coxnet", "gradient_boosting"],
#         help="Base estimator type"
#     )
#     parser.add_argument("--univariate_cox", action="store_true", help="Run univariate Cox selection before Stabl")
#     parser.add_argument("--univariate_p", type=float, default=0.05, help="P-value threshold for univariate Cox")

#     args = parser.parse_args()
    
#     # Setup logging
#     if args.debug_dir:
#         os.makedirs(args.debug_dir, exist_ok=True)
#         # Simple logging setup if needed
        
#     run_pipeline(args)

# if __name__ == "__main__":
#     main()
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import io
import traceback
import warnings

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
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# -------------------------------
# Utils (Ported from run_stabl_cox_FIXED.py)
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
    
    # [DEBUG] Check for duplicates
    if expr.index.duplicated().any():
        dup_genes = expr.index[expr.index.duplicated()].unique()
        print(f"[DEBUG] Found {len(dup_genes)} duplicated genes. First 50: {list(dup_genes)[:50]}")
        
        if debug_dir:
            try:
                os.makedirs(debug_dir, exist_ok=True)
                pd.Series(dup_genes, name="Duplicated_Genes").to_csv(Path(debug_dir) / "duplicated_genes.csv", index=False)
                print(f"[DEBUG] Saved duplicated genes list to {Path(debug_dir) / 'duplicated_genes.csv'}")
            except Exception as e:
                print(f"[DEBUG] Failed to save duplicated genes: {e}")
    else:
        print("[DEBUG] No duplicated genes found.")

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
    y = y.dropna(subset=["time", "event"])
    print("[DEBUG] y shape AFTER dropna:", y.shape)
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


def run_univariate_cox(X, y, p_thresh=0.05, debug_dir=None):
    """
    Run univariate Cox PH regression for each feature in X against y.
    Select features with p-value < p_thresh.
    """
    print(f"[INFO] Running univariate Cox selection (p < {p_thresh})...")
    try:
        from lifelines import CoxPHFitter
    except ImportError:
        print("[WARNING] lifelines not installed. Skipping univariate Cox selection.")
        print("          Please install it: pip install lifelines")
        return X

    genes = X.columns
    n_genes = len(genes)
    
    # Prepare data arrays once
    times = y['time'].values
    events = y['event'].values.astype(int) # lifelines prefers 0/1
    
    def fit_one_gene(gene, values):
        try:
            # Create mini dataframe
            df = pd.DataFrame({
                'feature': values,
                'T': times,
                'E': events
            })
            # Drop NaNs if any (though align_X_y should have handled it)
            df = df.dropna()
            
            if df['feature'].nunique() < 2:
                return (gene, 1.0) # Constant feature
                
            cph = CoxPHFitter()
            # suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cph.fit(df, duration_col='T', event_col='E')
            
            # Get p-value of the 'feature' coefficient
            pval = cph.summary.loc['feature', 'p']
            return (gene, pval)
        except Exception:
            return (gene, 1.0)

    # Run in parallel
    print(f"[INFO] Fitting {n_genes} univariate models...")
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(fit_one_gene)(g, X[g].values) 
        for g in genes
    )
    
    res_df = pd.DataFrame(results, columns=['gene', 'p_value'])
    
    # Save results
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        res_df.to_csv(Path(debug_dir) / "univariate_cox_results.csv", index=False)
        print(f"[DEBUG] Saved univariate p-values to {Path(debug_dir) / 'univariate_cox_results.csv'}")
        
    # Filter
    selected = res_df[res_df['p_value'] < p_thresh]['gene'].tolist()
    print(f"[INFO] Univariate Cox: {len(selected)}/{n_genes} features passed p < {p_thresh}")
    
    if len(selected) == 0:
        print("[WARNING] No features passed univariate filter! Reverting to all features.")
        return X
        
    return X[selected]


def build_stabl_cox(n_bootstraps=500, random_state=42, debug_dir=None, model_type="coxnet"):
    """
    ✅ v4 VERSION: Coxnet (Lasso), ComponentwiseGradientBoosting, or RSF for STABL feature selection.
    """
    if model_type == "coxnet":
        # Stabl will call utils.auto_mode_lambda_grid which now correctly uses Coxnet to find the path.
        lambda_grid = "auto"

        base = CoxnetSurvivalAnalysis(
            l1_ratio=0.9,
            fit_baseline_model=True,
            alpha_min_ratio=0.01,
            max_iter=1000000,
            tol=1e-7,
            verbose=False
        )
    elif model_type == "gradient_boosting":
        base = ComponentwiseGradientBoostingSurvivalAnalysis(
            loss="coxph",
            random_state=random_state,
            verbose=0
        )
        # Gradient Boosting uses n_estimators as the regularization parameter
        lambda_grid = {"n_estimators": np.arange(10, 200, 20)} 
        
    elif model_type == "rsf":
        base = RandomSurvivalForest(
            n_estimators=100,
            min_samples_split=10,
            max_depth=10,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
            verbose=0
        )
        lambda_grid = {"min_samples_leaf": np.arange(2, 22, 2)}

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    stabl_cox = Stabl(
        base_estimator=base,
        lambda_grid=lambda_grid,
        n_bootstraps=n_bootstraps,
        artificial_type="knockoff",
        artificial_proportion=1,
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

    if args.selected_features_file:
        print(f"\n[INFO] Skipping Selection. Loading features from {args.selected_features_file}...")
        try:
            # Try reading as CSV first
            sf_df = pd.read_csv(args.selected_features_file)
            # Assuming the first column contains the features, or a column named "Selected_Features"
            if "Selected_Features" in sf_df.columns:
                selected_features = sf_df["Selected_Features"].tolist()
            else:
                selected_features = sf_df.iloc[:, 0].tolist()
            
            print(f"[INFO] Loaded {len(selected_features)} features from file.")
        except Exception as e:
            print(f"[ERROR] Failed to load selected features from file: {e}")
            return
    else:
        # ---------------------------------------------------------
        # 1. Load Selection Data
        # ---------------------------------------------------------
        print("\n[STEP 1] Loading Selection Data...")
        X_sel = load_counts(args.selection_counts, num_genes=args.num_genes, debug_dir=args.debug_dir)
        y_sel = load_clinical(args.selection_clinical)
        X_sel, y_sel = align_X_y(X_sel, y_sel)
        
        # ---------------------------------------------------------
        # Preprocessing
        # ---------------------------------------------------------
        print(f"[INFO] Applying LowInfoFilter (max_nan_fraction=0.2)...")
        lif = LowInfoFilter(max_nan_fraction=0.2)
        lif.fit(X_sel)
        X_sel_np = lif.transform(X_sel)
        # Reconstruct DataFrame
        if hasattr(lif, "get_feature_names_out"):
                cols = lif.get_feature_names_out()
        else:
                # Fallback if get_feature_names_out not available or fails
                cols = X_sel.columns[lif.get_support()]
        
        X_sel = pd.DataFrame(X_sel_np, index=X_sel.index, columns=cols)
        print(f"       Features remaining: {X_sel.shape[1]}")

        print(f"[INFO] Applying SimpleImputer (median)...")
        imputer = SimpleImputer(strategy="median")
        X_sel_np = imputer.fit_transform(X_sel)
        X_sel = pd.DataFrame(X_sel_np, index=X_sel.index, columns=X_sel.columns)

        print(f"[INFO] Applying StandardScaler...")
        scaler = StandardScaler()
        X_sel_np = scaler.fit_transform(X_sel)
        X_sel = pd.DataFrame(X_sel_np, index=X_sel.index, columns=X_sel.columns)

        # ---------------------------------------------------------
        # DATA LEAKAGE CHECK (Added from FIXED version)
        # ---------------------------------------------------------
        leakage_keywords = [
            "os", "time", "overall_survival", "survival_time", "days_to_death", "days to death",
            "censored", "censor", "status", "event", "vital_status", "vital status",
            "outcome", "survival"
        ]
        to_drop = []
        for col in X_sel.columns:
            if str(col).strip().lower() in leakage_keywords:
                to_drop.append(col)
        
        if to_drop:
            print(f"[WARNING] Potential data leakage detected! Dropping columns from X_sel: {to_drop}")
            X_sel = X_sel.drop(columns=to_drop)
        # ---------------------------------------------------------

        print(f"Selection Data: {X_sel.shape[0]} samples, {X_sel.shape[1]} genes")
        
        # Check events-per-variable ratio
        n_events = y_sel['event'].sum()
        epv = n_events / X_sel.shape[1]
        print(f"[INFO] Selection Events: {n_events}/{X_sel.shape[0]} ({100*n_events/X_sel.shape[0]:.1f}%)")
        print(f"[INFO] Events-per-variable (EPV): {epv:.2f}")

        if epv < 10:
            print(f"[WARNING] EPV < 10 may lead to overfitting in selection. Consider reducing --num_genes.")

        # Univariate Cox Filter (Optional)
        if args.univariate_cox:
            X_sel = run_univariate_cox(X_sel, y_sel, p_thresh=args.univariate_p, debug_dir=args.debug_dir)

        # ---------------------------------------------------------
        # 2. Run STABL Feature Selection
        # ---------------------------------------------------------
        print("\n[STEP 2] Running STABL Feature Selection...")
        stabl_cox = build_stabl_cox(
            n_bootstraps=args.n_boot, 
            random_state=args.seed, 
            debug_dir=args.debug_dir,
            model_type=args.model_type
        )
        
        stabl_cox.fit(X_sel, y_sel)
        
        # Get selected features
        selected_mask = stabl_cox.get_support()
        if hasattr(stabl_cox, "feature_names_in_"):
            selected_features = stabl_cox.feature_names_in_[selected_mask]
        else:
            selected_features = X_sel.columns[selected_mask]
            
        print(f"[INFO] STABL selected {len(selected_features)} features: {list(selected_features)}")
        
        # Save selected features
        pd.Series(selected_features, name="Selected_Features").to_csv(
            Path(args.outdir) / "selected_features.csv", index=False
        )
        
        # Save STABL results (plots etc)
        try:
            save_stabl_results(
                stabl_cox, 
                Path(args.outdir), 
                X_sel, 
                y_sel, 
                task_type="survival"
            )
        except Exception as e:
            print(f"[WARNING] Failed to save STABL plots: {e}")

    if len(selected_features) == 0:
        print("[ERROR] No features selected. Cannot proceed to training.")
        return

    # ---------------------------------------------------------
    # 3. Load Verify Data and Split
    # ---------------------------------------------------------
    print("\n[STEP 3] Loading Verify Data and Splitting...")
    
    # Load Verify Data - Use same num_genes as selection if provided
    X_verify = load_counts(args.verify_counts, num_genes=args.num_genes, debug_dir=args.debug_dir) 
    y_verify = load_clinical(args.verify_clinical)
    X_verify, y_verify = align_X_y(X_verify, y_verify)
    
    print(f"Verify Data (Total): {X_verify.shape[0]} samples, {X_verify.shape[1]} genes")

    # Split Verify Data
    X_train, X_test, y_train, y_test = train_test_split(
        X_verify, y_verify, 
        test_size=args.verify_split_ratio, 
        random_state=args.seed,
        stratify=y_verify['event'] # Stratify by event status
    )
    
    print(f"Verify Train Split: {X_train.shape[0]} samples")
    print(f"Verify Test Split: {X_test.shape[0]} samples")

    # ---------------------------------------------------------
    # Preprocessing for Verification Data
    # ---------------------------------------------------------
    print("\n[INFO] Preprocessing Verify Data (Train/Test Split)...")
    
    # 1. For Full Features Evaluation
    # We work on copies to not affect the extraction of selected features later if we drop columns
    X_train_full = X_train.copy()
    X_test_full = X_test.copy()
    
    # LIF
    print(f"       [Full] Applying LowInfoFilter (max_nan_fraction=0.2)...")
    lif_full = LowInfoFilter(max_nan_fraction=0.2)
    lif_full.fit(X_train_full)
    
    # Transform and keep DataFrame
    cols_full = lif_full.get_feature_names_out() if hasattr(lif_full, "get_feature_names_out") else X_train_full.columns[lif_full.get_support()]
    X_train_full = pd.DataFrame(lif_full.transform(X_train_full), index=X_train_full.index, columns=cols_full)
    X_test_full = pd.DataFrame(lif_full.transform(X_test_full), index=X_test_full.index, columns=cols_full)
    
    # Imputer
    print(f"       [Full] Applying SimpleImputer (median)...")
    imputer_full = SimpleImputer(strategy="median")
    X_train_full = pd.DataFrame(imputer_full.fit_transform(X_train_full), index=X_train_full.index, columns=X_train_full.columns)
    X_test_full = pd.DataFrame(imputer_full.transform(X_test_full), index=X_test_full.index, columns=X_test_full.columns)

    # Scaler
    print(f"       [Full] Applying StandardScaler...")
    scaler_full = StandardScaler()
    X_train_full = pd.DataFrame(scaler_full.fit_transform(X_train_full), index=X_train_full.index, columns=X_train_full.columns)
    X_test_full = pd.DataFrame(scaler_full.transform(X_test_full), index=X_test_full.index, columns=X_test_full.columns)

    # 2. For Selected Features Evaluation
    # We start from the raw split again
    # Ensure all selected features exist
    missing_verify = set(selected_features) - set(X_train.columns)
    if missing_verify:
        print(f"[WARNING] {len(missing_verify)} selected features missing in Verify Data. Filling with 0.")
        for f in missing_verify:
            X_train[f] = 0.0
            X_test[f] = 0.0
            
    X_train_sub = X_train[selected_features].copy()
    X_test_sub = X_test[selected_features].copy()
    
    print(f"       [Selected] Applying LowInfoFilter (max_nan_fraction=0.2)...")
    lif_sub = LowInfoFilter(max_nan_fraction=0.2)
    lif_sub.fit(X_train_sub)
    
    cols_sub = lif_sub.get_feature_names_out() if hasattr(lif_sub, "get_feature_names_out") else X_train_sub.columns[lif_sub.get_support()]
    X_train_sub = pd.DataFrame(lif_sub.transform(X_train_sub), index=X_train_sub.index, columns=cols_sub)
    X_test_sub = pd.DataFrame(lif_sub.transform(X_test_sub), index=X_test_sub.index, columns=cols_sub)
    
    print(f"       [Selected] Features remaining after LIF: {X_train_sub.shape[1]}/{len(selected_features)}")

    # Imputer
    print(f"       [Selected] Applying SimpleImputer (median)...")
    imputer_sub = SimpleImputer(strategy="median")
    X_train_sub = pd.DataFrame(imputer_sub.fit_transform(X_train_sub), index=X_train_sub.index, columns=X_train_sub.columns)
    X_test_sub = pd.DataFrame(imputer_sub.transform(X_test_sub), index=X_test_sub.index, columns=X_test_sub.columns)

    # Scaler
    print(f"       [Selected] Applying StandardScaler...")
    scaler_sub = StandardScaler()
    X_train_sub = pd.DataFrame(scaler_sub.fit_transform(X_train_sub), index=X_train_sub.index, columns=X_train_sub.columns)
    X_test_sub = pd.DataFrame(scaler_sub.transform(X_test_sub), index=X_test_sub.index, columns=X_test_sub.columns)

    # ---------------------------------------------------------
    # 5. Train Final Models (Selected vs Full)
    # ---------------------------------------------------------
    print("\n[STEP 4] Training Final Models on Verify Train Split...")
    
    # Prepare structured arrays for sksurv
    y_train_surv = Surv.from_arrays(
        event=y_train['event'].astype(bool).values, 
        time=y_train['time'].values
    )
    y_test_surv = Surv.from_arrays(
        event=y_test['event'].astype(bool).values, 
        time=y_test['time'].values
    )

    # Define models to evaluate
    models_dict = {
        "CoxPH": CoxPHSurvivalAnalysis(),
        "Coxnet": CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01, fit_baseline_model=True),
        "GradientBoosting": ComponentwiseGradientBoostingSurvivalAnalysis(loss="coxph", random_state=args.seed),
        "RSF": RandomSurvivalForest(n_estimators=100, min_samples_split=10, min_samples_leaf=15, max_features="sqrt", n_jobs=-1, random_state=args.seed)
    }

    results_list = []

    def evaluate_models(X_tr, X_te, y_tr_s, y_te_s, label):
        print(f"\n--- Evaluating Models on {label} Features ({X_tr.shape[1]} features) ---")
        for name, model in models_dict.items():
            # Skip standard CoxPH if p > n (singular matrix issue)
            if name == "CoxPH" and X_tr.shape[1] > X_tr.shape[0]:
                print(f"Skipping {name} for {label} (p > n)")
                continue
                
            try:
                model.fit(X_tr, y_tr_s)
                c_train = model.score(X_tr, y_tr_s)
                c_test = model.score(X_te, y_te_s)
                print(f"[{label}] {name}: Train C={c_train:.4f}, Test C={c_test:.4f}")
                
                results_list.append({
                    "Feature_Set": label,
                    "Model": name,
                    "Train_C_Index": c_train,
                    "Test_C_Index": c_test,
                    "Num_Features": X_tr.shape[1]
                })
            except Exception as e:
                print(f"[{label}] {name} Failed: {e}")

    # 1. Evaluate on Selected Features
    evaluate_models(X_train_sub, X_test_sub, y_train_surv, y_test_surv, "Selected")

    # 2. Evaluate on Full Features (Verify Data)
    # Note: This might be slow if verify data has many genes
    evaluate_models(X_train_full, X_test_full, y_train_surv, y_test_surv, "Full")

    # ---------------------------------------------------------
    # 6. Save Results
    # ---------------------------------------------------------
    print("\n[STEP 5] Saving Results...")
    
    res_df = pd.DataFrame(results_list)
    res_df.to_csv(Path(args.outdir) / "final_model_comparison.csv", index=False)
    print(res_df)

    with open(Path(args.outdir) / "final_metrics.txt", "w") as f:
        f.write(f"Verify Total Samples: {X_verify.shape[0]}\n")
        f.write(f"Verify Train Samples: {X_train.shape[0]}\n")
        f.write(f"Verify Test Samples: {X_test.shape[0]}\n")
        f.write(f"Selected Features: {len(selected_features)}\n")
        f.write("\n--- Model Comparison ---\n")
        f.write(res_df.to_string())

    print(f"\n[DONE] Results saved to {args.outdir}")


def main(args):
    # Setup logging (Tee stdout/stderr)
    log_buffer = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)
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
            def fileno(self):
                return self.stream.fileno()
                
        sys.stdout = Tee(original_stdout, log_buffer)
        sys.stderr = Tee(original_stderr, log_buffer)

    try:
        run_pipeline(args)
    except KeyboardInterrupt:
        print("\n[INFO] Process interrupted by user (KeyboardInterrupt).")
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        traceback.print_exc()
    finally:
        if log_buffer:
            # Restore streams
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            
            # Save log
            try:
                log_path = Path(args.debug_dir) / "run_log.txt"
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(log_buffer.getvalue())
                print(f"[DEBUG] Terminal log saved to {log_path}")
            except Exception as e:
                print(f"[ERROR] Failed to save log file: {e}")


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
    parser.add_argument(
        "--num_genes",
        type=int,
        default=None,
        help="Keep only the first N genes in file order (for Selection data)."
    )
    parser.add_argument("--debug_dir", type=str, default=None, help="Directory to save debug info")
    parser.add_argument(
        "--model_type", 
        type=str, 
        default="coxnet", 
        choices=["coxnet", "gradient_boosting", "rsf"],
        help="Base estimator type"
    )
    parser.add_argument("--univariate_cox", action="store_true", help="Run univariate Cox selection before Stabl")
    parser.add_argument("--univariate_p", type=float, default=0.05, help="P-value threshold for univariate Cox")
    parser.add_argument("--selected_features_file", type=str, default=None, help="Path to file containing selected features (skip selection step)")

    args = parser.parse_args()
    main(args)