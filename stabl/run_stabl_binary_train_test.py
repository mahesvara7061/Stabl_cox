import pandas as pd
import numpy as np
import sys
import os
import argparse
import warnings
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import io
import traceback
import re

matplotlib.use('Agg')
import seaborn as sns

# Local imports
try:
    # Try importing as if installed or in pythonpath
    from stabl.stabl import Stabl
    from stabl.preprocessing import LowInfoFilter
except ImportError:
    # If running as script from stabl/ folder, we need to adjust sys.path
    # to import 'stabl' as a package from the parent directory.
    import sys
    from pathlib import Path
    
    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent
    
    # 1. Add parent dir to start of sys.path to prioritize package resolution
    sys.path.insert(0, str(parent_dir))
    
    # 2. Remove current dir from sys.path if present to avoid shadowing 'stabl' package with 'stabl.py'
    if str(current_dir) in sys.path:
        sys.path.remove(str(current_dir))
        
    try:
        from stabl.stabl import Stabl
        from stabl.preprocessing import LowInfoFilter
    except ImportError:
        # Restore path if failed, though unlikely to help if above failed
        sys.path.append(str(current_dir))
        raise


from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    average_precision_score,
    precision_recall_curve
)
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import mannwhitneyu, ttest_ind, uniform, loguniform, randint

# -------------------------------
# Utils
# -------------------------------

class PLSDA(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pls = None

    def fit(self, X, y):
        # Map binary 0/1 to -1/1 or similar for regression if needed, 
        # but PLS works with 0/1. Sklearn PLSRegression is multi-output capable.
        self.pls = PLSRegression(n_components=self.n_components)
        self.pls.fit(X, y)
        return self

    def predict(self, X):
        pred_scores = self.pls.predict(X)
        # Threshold at 0.5 for 0/1 encoding
        return (pred_scores >= 0.5).astype(int).ravel()

    def predict_proba(self, X):
        # PLS doesn't give probabilities naturally. 
        # We'll return the regression scores normalized vaguely or just raw scores for ranking
        # ROC AUC uses rank, so raw regression score is fine usually.
        # But sklearn expects probability-like (0-1).
        # We will clip to 0-1 for simplicity or use logistic sigmoid.
        pred_scores = self.pls.predict(X).ravel()
        # Simple sigmoid to squash to 0-1
        probs = 1 / (1 + np.exp(-pred_scores)) 
        return np.vstack([1-probs, probs]).T

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
                dup_genes = expr.index[expr.index.duplicated()].unique()
                with open(Path(debug_dir) / "duplicate_genes.txt", "w") as f:
                    for g in dup_genes:
                        f.write(f"{g}\n")
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


def load_clinical(clinical_path):
    """
    Read clinical with columns: sample id, prognosis.
    Labels: "good", "poor" (case-insensitive).
    Mapped to: 0 (Good/Control), 1 (Poor/Case).
    """
    sep = detect_sep(clinical_path)
    clin = pd.read_csv(clinical_path, sep=sep, header=0)
    clin.columns = [c.replace("\ufeff", "") for c in clin.columns]

    ren = {c: c.strip().lower() for c in clin.columns}
    clin = clin.rename(columns=ren)

    col_map = {}
    for cand in ["sample id", "sample_id", "sample", "id", "case_id", "patient_id", "case submitter id", "cgga_id"]:
        if cand in clin.columns:
            col_map["sample_id"] = cand
            break
            
    # Try to find prognosis column
    for cand in ["prognosis", "label", "group", "class", "prognosis_binary", "response", "cluster", "risk"]:
         if cand in clin.columns:
            col_map["target"] = cand
            break

    if set(col_map.keys()) != {"sample_id", "target"}:
        raise ValueError(f"Cannot find all required columns (sample_id, prognosis). Found: {list(clin.columns)}")

    sid_raw = clin[col_map["sample_id"]].astype(str)
    target_raw = clin[col_map["target"]].astype(str).str.strip().str.lower()

    # Define Mapping
    label_map = {
        "poor": 1, "bad": 1, "high": 1, "recurrence": 1, "dead": 1, "1": 1, "1.0": 1, "case": 1,
        "good": 0, "low": 0, "no_recurrence": 0, "alive": 0, "0": 0, "0.0": 0, "control": 0
    }
    
    y_mapped = target_raw.map(label_map)
    
    # Check if any NaNs after mapping
    if y_mapped.isna().any():
        unmapped = target_raw[y_mapped.isna()].unique()
        print(f"[WARN] The following labels could not be mapped automatically: {unmapped}. Rows will be dropped.")
        
    y_mapped = y_mapped.dropna()
    y = y_mapped.astype(int)
    
    # Set index
    y.index = pd.Index(sid_raw[y_mapped.index].to_numpy(), name="sample_id")

    if y.index.duplicated().any():
        y = y[~y.index.duplicated(keep="first")]

    print(f"[INFO] Loaded clinical data with {len(y)} samples.")
    print(f"       Class distribution: \n{y.value_counts()}")
    return y


def align_X_y(X, y):
    inter = X.index.astype(str).intersection(y.index.astype(str))
    X2 = X.loc[inter].sort_index()
    y2 = y.loc[inter].sort_index()
    return X2, y2


def run_univariate_analysis(X, y, p_thresh=0.05, debug_dir=None):
    """
    Run Mann-Whitney U test (or t-test) for each feature.
    """
    print(f"[INFO] Running univariate selection (Mann-Whitney U, p < {p_thresh})...")
    
    genes = X.columns
    # Groups
    group0 = y[y == 0].index
    group1 = y[y == 1].index
    
    def test_one_gene(gene, values, g0_idx, g1_idx):
        try:
            val0 = values.loc[g0_idx].dropna()
            val1 = values.loc[g1_idx].dropna()
            if len(val0) < 2 or len(val1) < 2:
                return (gene, 1.0)
            
            # Mann-Whitney U test
            stat, p = mannwhitneyu(val0, val1, alternative='two-sided')
            return (gene, p)
        except Exception:
            return (gene, 1.0)

    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(test_one_gene)(g, X[g], group0, group1) for g in genes
    )
    res_df = pd.DataFrame(results, columns=['gene', 'p_value'])
    
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        res_df.to_csv(Path(debug_dir) / "univariate_results.csv", index=False)
        
    selected = res_df[res_df['p_value'] < p_thresh]['gene'].tolist()
    print(f"[INFO] Univariate: {len(selected)}/{len(genes)} features passed p < {p_thresh}")
    if len(selected) == 0: return X
    return X[selected]

# -------------------------------
# NEW: Comprehensive Evaluation Functions for Classification
# -------------------------------

def run_nested_cv_evaluation(X, y, estimators, outdir, dataset_name="Verify", n_outer=5, n_inner=5, seed=42):
    """
    Perform Nested CV (Outer StratifiedKFold, Inner GridSearchCV) for multiple estimators.
    Calculates AUC, ROC curves.
    """
    print(f"\n[EVALUATION] Starting Nested CV on {dataset_name} ({X.shape[1]} features, {X.shape[0]} samples)...")
    
    results_stats = []
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Outer CV
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    # Common grid for ROC interpolation
    mean_fpr = np.linspace(0, 1, 100)
    
    # Pre-calculated splits to ensure same folds for all models if X is same
    # But X might change (Full vs Selected), so we must trust random_state
    
    # Loop over models
    for name, (base_model, param_grid) in estimators.items():
        print(f"   >> Evaluating {name}...")
        
        auc_scores = []
        tprs = []
        best_params_list = []
        
        for i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # --- Preprocessing inside fold ---
            # (Important for valid CV: fit scaler/imputer on training fold only)
            imputer = SimpleImputer(strategy="median")
            X_train_imp = imputer.fit_transform(X_train)
            X_test_imp = imputer.transform(X_test)
            
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train_imp)
            X_test_sc = scaler.transform(X_test_imp)
            
            # --- Inner CV for Hyperparameter Tuning ---
            inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed + i)
            
            # Using RandomizedSearchCV for tuning
            clf = RandomizedSearchCV(
                base_model, 
                param_distributions=param_grid, 
                n_iter=20,  # Number of parameter settings that are sampled
                cv=inner_cv, 
                scoring='roc_auc', 
                n_jobs=-1,
                random_state=seed + i
            )
            clf.fit(X_train_sc, y_train)
            
            best_model = clf.best_estimator_
            best_params_list.append(clf.best_params_)
            
            # --- Prediction on Outer Test ---
            if hasattr(best_model, "predict_proba"):
                y_prob = best_model.predict_proba(X_test_sc)[:, 1]
            elif hasattr(best_model, "decision_function"):
                y_prob = best_model.decision_function(X_test_sc)
            else:
                # e.g. PLSDA custom
                y_prob = best_model.predict(X_test_sc) # Attempt fallback

            # --- Metrics ---
            try:
                fold_auc = roc_auc_score(y_test, y_prob)
            except ValueError:
                fold_auc = np.nan
            auc_scores.append(fold_auc)
            
            # ROC Curve Interpolation
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            
        # --- Aggregation for this Model ---
        mean_auc = np.nanmean(auc_scores)
        std_auc = np.nanstd(auc_scores)
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        
        # Save Stats
        results_stats.append({
            "Dataset": dataset_name,
            "Model": name,
            "Mean_AUC": mean_auc,
            "Std_AUC": std_auc,
            "Best_Params_Fold0": str(best_params_list[0]) # just example
        })
        
        # Plot Mean ROC
        plt.figure(figsize=(8, 6))
        
        # Plot individual folds
        for fit_idx, tpr_fold in enumerate(tprs):
            plt.plot(mean_fpr, tpr_fold, lw=1, alpha=0.3, label=f'Fold {fit_idx+1}')
            
        plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})', lw=2, alpha=.8)
        
        # Add Standard Deviation Shade
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
        
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{name} - ROC Curve ({dataset_name})')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(out_path / f"NestedCV_ROC_{dataset_name}_{name}.png")
        plt.close()
        
    return pd.DataFrame(results_stats)

def analyze_individual_genes_binary(X, y, selected_features, outdir, dataset_label=""):
    """
    Detailed analysis of selected genes for binary classification.
    1. Univariate Test (Mann-Whitney U)
    2. Boxplots (Good vs Poor)
    """
    label_str = f" ({dataset_label})" if dataset_label else ""
    print(f"\n[INFO] Analyzing individual genes{label_str} for {len(selected_features)} selected features...")
    
    folder_name = "individual_genes_analysis"
    csv_name = "univariate_analysis_selected_genes.csv"
    
    if dataset_label:
        folder_name += f"_{dataset_label}"
        csv_name = re.sub(r"\.csv$", f"_{dataset_label}.csv", csv_name)

    out_path = Path(outdir) / folder_name
    out_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    available_feats = [f for f in selected_features if f in X.columns]
    if len(available_feats) < len(selected_features):
        print(f"[WARN] {len(selected_features)-len(available_feats)} features missing in dataset.")
    
    group0 = y == 0
    group1 = y == 1
    
    for gene in available_feats:
        # Stats
        val0 = X[gene][group0].dropna()
        val1 = X[gene][group1].dropna()
        
        try:
            stat, p = mannwhitneyu(val0, val1)
            mean0 = val0.mean()
            mean1 = val1.mean()
            fc = mean1 - mean0 # or log2 fold change if log data
        except Exception:
            p = 1.0
            fc = 0.0

        results.append({
            "gene": gene,
            "p_value": p,
            "mean_class0": mean0 if 'mean0' in locals() else np.nan,
            "mean_class1": mean1 if 'mean1' in locals() else np.nan,
            "diff_means": fc
        })

        # Boxplot
        try:
            plt.figure(figsize=(6, 5))
            data_plot = pd.DataFrame({
                "Expression": X[gene],
                "Group": y.map({0: "Good/Control", 1: "Poor/Case"})
            })
            sns.boxplot(x="Group", y="Expression", hue="Group", data=data_plot, palette="Set2", legend=False)
            sns.stripplot(x="Group", y="Expression", data=data_plot, color='black', alpha=0.3, jitter=True)
            plt.title(f"{gene} (p={p:.2e})")
            plt.savefig(out_path / f"Boxplot_{gene}.png")
            plt.close()
        except Exception as e:
            print(f"Error plotting {gene}: {e}")

    if results:
        res_df = pd.DataFrame(results)
        res_df = res_df.sort_values("p_value")
        save_path = Path(outdir) / csv_name
        res_df.to_csv(save_path, index=False)
        print(res_df.to_string())


# -------------------------------
# Main Pipeline
# -------------------------------

def build_stabl_binary(n_bootstraps=500, random_state=42, debug_dir=None, model_type="logistic"):
    lambda_grid = None
    if model_type == "logistic":
        # Standard Logistic Regression with L1 penalty
        # Aligned with run_val_COVID.py
        base = LogisticRegression(
            penalty='l1', 
            solver='liblinear', 
            class_weight='balanced', 
            max_iter=int(1e6),
            random_state=random_state
        )
        lambda_grid = {"C": np.linspace(0.01, 1, 30)}
    elif model_type == "rf":
        # Random Forest: Vary complexity via min_samples_leaf
        base = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced')
        lambda_grid = {"min_samples_leaf": [1, 2, 4, 6, 8, 10, 15, 20]}
    elif model_type == "gbm":
        # GBM: Vary n_estimators (early stopping effect via stability)
        base = GradientBoostingClassifier(random_state=random_state)
        lambda_grid = {"n_estimators": np.arange(50, 300, 25)}
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # For Logistic Regression, lambda_grid is handled by Stabl if passed as "auto" or dict
    # Default Stabl constructor uses LogisticRegression(penalty='l1') 
    
    stabl = Stabl(
        base_estimator=base,
        n_bootstraps=n_bootstraps,
        artificial_type="knockoff",
        artificial_proportion=1.0,
        sample_fraction=0.5,
        replace=False,
        fdr_threshold_range=np.arange(0.1, 1, 0.01),
        lambda_grid=lambda_grid,
        explore=True,
        n_explore=5,
        task_type="binary",
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        debug_dir=debug_dir
    )
    return stabl


def run_pipeline(args):
    os.makedirs(args.outdir, exist_ok=True)
    selected_features = []

    # ---------------------------------------------------------
    # 1. Load Data & Align Genes
    # ---------------------------------------------------------
    print("\n[STEP 1] Loading Data & Aligning Genes...")
    
    # Load Selection
    print(f"Loading Selection Data: {args.selection_counts}")
    X_sel = load_counts(args.selection_counts, num_genes=args.num_genes, debug_dir=args.debug_dir)
    y_sel = load_clinical(args.selection_clinical)
    X_sel, y_sel = align_X_y(X_sel, y_sel)
    print(f"  Selection Data (Raw): {X_sel.shape[0]} samples, {X_sel.shape[1]} genes")

    # Load Verify
    print(f"Loading Verify Data: {args.verify_counts}")
    X_verify = load_counts(args.verify_counts, num_genes=args.num_genes, debug_dir=args.debug_dir) 
    y_verify = load_clinical(args.verify_clinical)
    X_verify, y_verify = align_X_y(X_verify, y_verify)
    print(f"  Verify Data (Raw): {X_verify.shape[0]} samples, {X_verify.shape[1]} genes")
    
    # Intersect Genes
    common_genes = X_sel.columns.intersection(X_verify.columns)
    print(f"[INFO] Intersecting genes: {len(X_sel.columns)} (Sel) & {len(X_verify.columns)} (Ver) -> {len(common_genes)} Common Genes")
    
    if len(common_genes) == 0:
        raise ValueError("No common genes found between Selection and Verify datasets!")
    
    X_sel = X_sel[common_genes]
    X_verify = X_verify[common_genes] # Keep verify aligned but raw/imputed later
    
    # Preprocessing Selection Data for STABL
    print(f"[INFO] Preprocessing Selection Data (LowInfoFilter, Impute, Scale)...")
    lif = LowInfoFilter(max_nan_fraction=0.2)
    lif.fit(X_sel)
    X_sel_np = lif.transform(X_sel)
    cols = lif.get_feature_names_out() if hasattr(lif, "get_feature_names_out") else X_sel.columns[lif.get_support()]
    X_sel = pd.DataFrame(X_sel_np, index=X_sel.index, columns=cols)
    
    imputer = SimpleImputer(strategy="median")
    X_sel = pd.DataFrame(imputer.fit_transform(X_sel), index=X_sel.index, columns=X_sel.columns)
    scaler = StandardScaler()
    X_sel = pd.DataFrame(scaler.fit_transform(X_sel), index=X_sel.index, columns=X_sel.columns)

    if args.selected_features_file:
        print(f"\n[INFO] Skipping Selection. Loading features from {args.selected_features_file}...")
        try:
             # Try reading as csv with 'Selected_Features' or just a list
            temp_df = pd.read_csv(args.selected_features_file)
            if "Selected_Features" in temp_df.columns:
                selected_features = temp_df["Selected_Features"].tolist()
            else:
                # Assume first column
                selected_features = temp_df.iloc[:, 0].tolist()
        except Exception as e:
            print(f"Error reading selected features: {e}. Trying simple read.")
            with open(args.selected_features_file, 'r') as f:
                selected_features = [line.strip() for line in f if line.strip()] 
    else:
        if args.univariate:
            X_sel = run_univariate_analysis(X_sel, y_sel, p_thresh=args.univariate_p, debug_dir=args.debug_dir)

        # ---------------------------------------------------------
        # 2. Run STABL Feature Selection
        # ---------------------------------------------------------
        print("\n[STEP 2] Running STABL Feature Selection (Binary)...")
        stabl = build_stabl_binary(n_bootstraps=args.n_boot, random_state=args.seed, debug_dir=args.debug_dir, model_type=args.model_type)
        stabl.fit(X_sel, y_sel)
        
        selected_mask = stabl.get_support()
        try:
             # Try standard sklearn way
             if hasattr(stabl, "feature_names_in_"):
                 selected_features = stabl.feature_names_in_[selected_mask].tolist()
             else:
                 selected_features = X_sel.columns[selected_mask].tolist()
        except:
             selected_features = X_sel.columns[selected_mask].tolist()

        print(f"[INFO] STABL selected {len(selected_features)} features: {list(selected_features)}")
        pd.Series(selected_features, name="Selected_Features").to_csv(Path(args.outdir) / "selected_features.csv", index=False)
        try:
            stabl.plot_fdr_graph(Path(args.outdir))
        except Exception: 
            pass

    if len(selected_features) == 0:
        print("[ERROR] No features selected. Cannot proceed to training.")
        return

    # --- Analysis on Selection Data ---
    print("\n[ANALYSIS] Analyzing Individual Genes on Selection Dataset...")
    analyze_individual_genes_binary(X_sel, y_sel, selected_features, args.outdir, dataset_label="selection_dataset")

    # ---------------------------------------------------------
    # 3. Use Pre-loaded Verify Data
    # ---------------------------------------------------------
    print("\n[STEP 3] Using Pre-loaded Verify Data...")
    # Already loaded and aligned in Step 1
    # X_verify, y_verify are ready
    print(f"  Verify Data: {X_verify.shape[0]} samples, {X_verify.shape[1]} genes")

    print("\n[ANALYSIS] Analyzing Individual Genes on Entire Verify Dataset...")
    X_verify_analyze = X_verify.copy()
    
    missing_feats = [f for f in selected_features if f not in X_verify_analyze.columns]
    if missing_feats:
        print(f"[WARN] {len(missing_feats)} features missing in Verify dataset. Filling with 0.")
    
    # Fill missing features with median or 0
    # Note: X_verify contains all genes. We just need to ensure selected_features exist.
    for f in missing_feats:
        X_verify_analyze[f] = 0.0

    # Ensure we have all selected features for analysis
    # Analysis using ONLY selected features on full set
    X_verify_sel_only = X_verify_analyze.reindex(columns=selected_features).fillna(0)
    
    analyze_individual_genes_binary(X_verify_sel_only, y_verify, selected_features, args.outdir, dataset_label="verify_entire_dataset")

    # ---------------------------------------------------------
    # 5. Nested CV Evaluation
    # ---------------------------------------------------------
    print("\n[STEP 5] Running Nested CV Evaluation on Verify Data...")

    # Define Estimators and Grids (Distributions for RandomizedSearchCV)
    estimators = {
        "LogisticRegression": (
            LogisticRegression(max_iter=5000, class_weight='balanced', solver='liblinear', random_state=args.seed),
            {
                "C": loguniform(1e-2, 1e2), 
                "penalty": ["l1", "l2"]
            }
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=args.seed, class_weight='balanced'),
            {
                "n_estimators": randint(100, 500), 
                "max_depth": [None, 5, 10, 20], 
                "min_samples_leaf": randint(1, 10),
                "max_features": ["sqrt", "log2", 0.5]
            }
        ),
        "SVM": (
            SVC(probability=True, class_weight='balanced', random_state=args.seed),
            {
                "C": loguniform(1e-2, 1e2), 
                "kernel": ["linear", "rbf"], 
                "gamma": ["scale", "auto"] + list(np.logspace(-3, 0, 10))
            }
        ),
    }

    results_all = []

    # A. Evaluation on SELECTED Features
    # Ensure X_verify contains all selected features (impute if missing)
    X_verify_sel = X_verify.reindex(columns=selected_features).fillna(0)
    
    print(f"\n--- Nested CV on SELECTED Features ({X_verify_sel.shape[1]}) ---")
    res_sel = run_nested_cv_evaluation(X_verify_sel, y_verify, estimators, args.outdir, dataset_name="Selected_Features", seed=args.seed)
    res_sel["Features"] = "Selected"
    results_all.append(res_sel)

    # B. Evaluation on FULL Features
    # Use LIF to reduce dimensionality slightly if huge? Or use all?
    # Usually 'Full Genes' means all genes available after basic filtering.
    # We apply basic LowInfoFilter to avoid constant columns in CV
    print(f"\n--- Nested CV on FULL Features ---")
    lif = LowInfoFilter(max_nan_fraction=0.2)
    X_verify_full_np = lif.fit_transform(X_verify)
    cols = lif.get_feature_names_out() if hasattr(lif, "get_feature_names_out") else X_verify.columns[lif.get_support()]
    X_verify_full = pd.DataFrame(X_verify_full_np, index=X_verify.index, columns=cols)
    # Simple median imputation for full set
    X_verify_full = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(X_verify_full), index=X_verify_full.index, columns=X_verify_full.columns)
    
    print(f"    Full Features count: {X_verify_full.shape[1]}")
    res_full = run_nested_cv_evaluation(X_verify_full, y_verify, estimators, args.outdir, dataset_name="Full_Features", seed=args.seed)
    res_full["Features"] = "Full"
    results_all.append(res_full)

    # ---------------------------------------------------------
    # 6. Save Final Results
    # ---------------------------------------------------------
    print("\n[STEP 6] Saving Final Comparison...")
    final_df = pd.concat(results_all, ignore_index=True)
    final_df.to_csv(Path(args.outdir) / "final_model_comparison_nested_cv.csv", index=False)
    print(final_df)
    print(f"\n[DONE] Results saved to {args.outdir}")


def main(args):
    log_buffer = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)
        log_buffer = io.StringIO()
        class Tee(object):
            def __init__(self, *files):
                self.files = files
            def write(self, obj):
                for f in self.files:
                    f.write(obj)
                    f.flush()
            def flush(self):
                for f in self.files:
                    f.flush()
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
        if log_buffer and args.debug_dir:
            with open(Path(args.debug_dir) / "run_log.txt", "w") as f:
                f.write(log_buffer.getvalue())
        sys.stdout = original_stdout
        sys.stderr = original_stderr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run STABL Binary Classification: Selection on one dataset, Verify on another (split internally)")
    
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
    
    parser.add_argument("--model_type", type=str, default="logistic", choices=["logistic", "rf", "gbm"], help="Base estimator type for STABL")
    
    parser.add_argument("--univariate", action="store_true", help="Run univariate selection (Mann-Whitney) before Stabl")
    parser.add_argument("--univariate_p", type=float, default=0.05, help="P-value threshold for univariate selection")
    parser.add_argument("--selected_features_file", type=str, default=None, help="Path to file containing selected features (skip selection step)")

    args = parser.parse_args()
    main(args)
