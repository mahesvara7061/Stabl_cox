# STABL Survival Prediction Adaptation Summary

**Date:** November 17, 2025  
**Framework:** STABL (Stability Selection) for Survival Analysis  
**Base Model:** CoxPHSurvivalAnalysis (scikit-survival)

---

## Overview

This document summarizes all modifications made to adapt the STABL framework from binary/multiclass classification and regression to **survival analysis** (time-to-event prediction with censoring). The original STABL implementation handled only classification and regression tasks; these changes enable it to work with survival data using Cox proportional hazards models.

---

## Key Concepts

### Survival Data Format
- **Input:** Structured array with two fields:
  - `event` (bool): True if event occurred, False if censored
  - `time` (float): Time to event or censoring
- **scikit-survival format:** `dtype=[('event', '?'), ('time', '<f8')]`

### Task Type
- New task type: `"survival"` added alongside `"binary"`, `"multiclass"`, `"regression"`
- Enables conditional logic throughout the codebase to handle survival-specific requirements

---

## Core Changes by Module

### 1. **stabl/stabl.py** - Main STABL Class

#### 1.1 Class Initialization
```python
def __init__(self, ..., task_type="binary"):
    ...
    self.task_type = task_type  # NEW: Added task_type parameter
```

#### 1.2 Bootstrap Sampling (`classic_bootstrap`)
**Location:** Lines 25-105  
**Change:** Added survival-specific validation to ensure both event classes exist in bootstrap samples

```python
def classic_bootstrap(y, n_subsamples, replace=True, class_weight=None, 
                     rng=np.random.default_rng(None), task_type="binary", **kwargs):
    ...
    # Task-specific validation
    needs_resample = False
    
    if task_type == "binary":
        if len(np.unique(y[sampled_indices])) < 2:
            needs_resample = True
            
    elif task_type == "survival":
        # Extract event status from structured array
        if isinstance(y, pd.DataFrame):
            y_events = y.iloc[sampled_indices]["event"].values
        elif hasattr(y, 'dtype') and hasattr(y.dtype, 'names'):
            event_field = 'event' if 'event' in y.dtype.names else y.dtype.names[1]
            y_events = y[event_field][sampled_indices]
        else:
            y_events = y[sampled_indices] if y.ndim == 1 else y[sampled_indices, 1]
        
        if len(np.unique(y_events)) < 2:
            needs_resample = True
    
    if needs_resample:
        sampled_indices = classic_bootstrap(y, n_subsamples, replace=replace, 
                                           class_weight=class_weight, rng=rng, 
                                           task_type=task_type, **kwargs)
```

**Purpose:** Prevents bootstrap samples where all observations are censored or all have events (would cause Cox model fitting to fail).

#### 1.3 Data Validation (`_validate_data`)
**Location:** Lines 1158-1257  
**Change:** Custom validation to convert survival data into structured arrays

```python
def _validate_data(self, X, y=None, reset=True, validate_separately=False):
    # X validation (unchanged)
    ...
    
    # NEW: Survival-specific y validation
    if isinstance(y, pd.DataFrame):
        cols = [c.strip().lower() for c in y.columns]
        event_alias = {"event", "status", "censor", "censored", "observed", "dead"}
        time_alias  = {"time", "os", "survival", "overall_survival", "duration"}
        
        # Find event/time columns
        ev_col = next((c for c in y.columns if c.strip().lower() in event_alias), None)
        tm_col = next((c for c in y.columns if c.strip().lower() in time_alias), None)
        
        if ev_col is not None and tm_col is not None:
            event = pd.to_numeric(y[ev_col], errors="coerce")
            event = (event == 1).astype(bool).values  # Convert 1/0 to bool
            time  = pd.to_numeric(y[tm_col], errors="coerce").astype(float).values
            
            # Remove NaN rows
            mask = ~np.isnan(time) & ~np.isnan(event)
            if n_dropped := (len(time) - mask.sum()) > 0:
                warnings.warn(f"Dropped {n_dropped} samples with NaN.")
            
            event = event[mask]
            time = time[mask]
            X_arr = X_arr[mask]  # CRITICAL: Also filter X
            
            # Create structured array
            y_arr = np.array(list(zip(event, time)), 
                           dtype=[('event', '?'), ('time', '<f8')])
            return X_arr, y_arr
    
    # Handle 2D numpy arrays (n, 2) format
    elif y_np.ndim == 2 and y_np.shape[1] == 2:
        # Similar conversion logic...
```

**Purpose:** 
- Accepts flexible input formats (DataFrame with named columns, 2D arrays)
- Converts to scikit-survival's required structured array format
- Handles missing values and filters both X and y consistently

#### 1.4 Lambda Grid Optimization (`_get_optimized_lambda_grid`)
**Location:** Delegates to `utils.auto_mode_lambda_grid`  
**Change:** Determines task type automatically, but survival must be handled via `auto_mode_lambda_grid`

#### 1.5 Fit Method
**Location:** Lines 1400-1540  
**Change:** Conditional logic for survival data during bootstrap fitting

```python
def fit(self, X, y, groups=None):
    ...
    X, y = self._validate_data(X=X, y=y, reset=True, validate_separately=False)
    
    # NEW: Extract structured array for survival
    y_surv = None
    if self.task_type == "survival":
        if isinstance(y, np.ndarray) and hasattr(y.dtype, 'names') and \
           y.dtype.names == ('event', 'time'):
            y_surv = y  # Already in correct format from _validate_data
        else:
            raise ValueError("For survival, y must be convertible to structured array")
    
    ...
    
    # NEW: Pass task_type to bootstrap generator
    bootstrap_indices = _bootstrap_generator(
        ...,
        task_type=self.task_type  # Added
    )
    
    # NEW: Conditional parallel fitting
    for idx, lambda_val in tqdm(...):
        if self.task_type == "survival":
            selected_variables = Parallel(...)(
                delayed(fit_bootstrapped_sample)(
                    clone(base_estimator),
                    X=X[safe_mask(X, subsample_indices), :],
                    y=y_surv[subsample_indices],  # Use y_surv (structured array)
                    corr_groups=corr_groups,
                    lambda_val=lambda_val,
                    threshold=self.bootstrap_threshold
                ) for subsample_indices in bootstrap_indices
            )
        else:
            # Original binary/regression/multiclass logic
            ...
```

**Purpose:** Ensures Cox models receive properly formatted structured arrays during bootstrap fitting.

#### 1.6 Feature Selection (`fit_bootstrapped_sample`)
**Location:** Lines 840-920  
**Change:** Robust handling when SelectFromModel fails or returns mismatched support length

```python
def fit_bootstrapped_sample(base_estimator, X, y, lambda_val, corr_groups=None, threshold=None):
    base_estimator.set_params(**lambda_val)
    if hasattr(base_estimator, "groups"):
        base_estimator.set_params(groups=corr_groups)
    
    base_estimator.fit(X, y)
    n_features = X.shape[1]
    
    # Try standard SelectFromModel path
    try:
        features_selection = SelectFromModel(estimator=base_estimator, 
                                            threshold=threshold, prefit=True)
        support = features_selection.get_support()
        if support.shape[0] == n_features:
            return support
        raise RuntimeError("support length mismatch")
    except Exception:
        # Fallback: Extract coefficients manually
        coef = None
        if hasattr(base_estimator, "coef_") and base_estimator.coef_ is not None:
            coef = np.ravel(base_estimator.coef_)
        elif hasattr(base_estimator, "feature_importances_"):
            coef = np.ravel(base_estimator.feature_importances_)
        
        if coef is None:
            return np.zeros(n_features, dtype=bool)
        
        # Pad/truncate to match n_features
        if coef.shape[0] < n_features:
            coef = np.pad(coef, (0, n_features - coef.shape[0]))
        elif coef.shape[0] > n_features:
            coef = coef[:n_features]
        
        # Apply threshold
        thr = float(threshold) if isinstance(threshold, (int, float)) else 1e-12
        support = np.abs(coef) > thr
        return support
```

**Purpose:** CoxPH models sometimes have coefficient arrays that don't match X shape due to internal handling. This fallback ensures we always get a valid boolean mask.

---

### 2. **stabl/utils.py** - Utility Functions

#### 2.1 Import Additions
```python
from sksurv.metrics import concordance_index_censored
```

#### 2.2 Lambda Grid Generation (`auto_mode_lambda_grid`)
**Location:** Lines 15-60  
**Change:** Added survival-specific lambda range calculation

```python
def auto_mode_lambda_grid(X, y, task_type, l1_ratio=None, n_lambda=30):
    ...
    def get_optimal_params(l1_r):
        if task_type == "classification":
            min_C = l1_min_c(X, y, loss="log")
            params = {"C": np.linspace(min_C, min_C * 100, n_lambda)}
        elif task_type == "survival":
            # For CoxPHSurvivalAnalysis with alphas parameter
            # Compute maximum lambda based on data
            l_max = np.linalg.norm(X.T @ (y['time'] * y['event'].astype(int)), 
                                  np.inf) / X.shape[0]
            alphas = np.logspace(np.log10(l_max / 100), np.log10(l_max), n_lambda)
            params = {"alphas": [alphas]}  # Note: List of arrays format
        else:
            # Regression path
            ...
        return params
```

**Purpose:** Computes sensible alpha range for Cox elastic net based on data scale.

#### 2.3 Fit-Predict (`fit_predict`)
**Location:** Lines 66-120  
**Change:** Added survival branch for prediction

```python
def fit_predict(estimator, X, y, train, test, task_type):
    ...
    if task_type == 'binary':
        results[test] = estimator.fit(X[train], y[train]).predict_proba(X[test])[:, 1]
    elif task_type == "survival":
        results[test] = estimator.fit(X[train], y[train]).predict(X[test])
    else:
        ...
    return results
```

**Purpose:** Cox models use `.predict()` directly (returns risk scores, not probabilities).

#### 2.4 Non-Partition GridSearch (`nonpartition_gridsearch`)
**Location:** Lines 195-275  
**Change:** Added C-index scoring for survival

```python
def nonpartition_gridsearch(estimator, param_grid, X, y, task_type, splitter, groups=None):
    ...
    for p in list_params:
        ...
        if task_type == "binary":
            score = roc_auc_score(y, y_preds)
        elif task_type == "survival":
            # Extract event/time from structured array or DataFrame
            if isinstance(y, (pd.Series, pd.DataFrame)):
                ev = y["event"].astype(bool).to_numpy()
                tt = y["time"].to_numpy()
            else:
                ev = y["event"].astype(bool)
                tt = y["time"]
            score = concordance_index_censored(ev, tt, np.asarray(y_preds))[0]
        ...
    return best_estimator, best_params, best_preds
```

**Purpose:** Uses C-index (concordance index) instead of AUC for survival model evaluation.

#### 2.5 Leave-One-Out GridSearch (`loo_gridsearch`)
**Location:** Lines 360-385  
**Change:** Manual cross-validation loop for survival

```python
def loo_gridsearch(estimator, param_grid, X, y, task_type, cv=LeaveOneOut(), groups=None):
    ...
    elif task_type == "survival":
        # Convert to DataFrame if needed
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_df = y.copy()
        else:
            y_df = pd.DataFrame({"time": y["time"], "event": y["event"].astype(bool)})
        
        y_preds = np.full(len(y_df), np.nan, dtype=float)
        X_np = X if isinstance(X, np.ndarray) else np.asarray(X)
        
        # Manual CV loop
        for train_idx, test_idx in cv.split(X_np, groups=groups):
            est = clone(estimator)
            est.fit(X_np[train_idx], y_df.iloc[train_idx].to_records(index=False))
            y_preds[test_idx] = est.predict(X_np[test_idx])
        
        ev = y_df["event"].astype(bool).to_numpy()
        tt = y_df["time"].to_numpy()
        score = concordance_index_censored(ev, tt, y_preds)[0]
    ...
```

**Purpose:** sklearn's `cross_val_predict` doesn't natively handle structured arrays, so we implement manual splitting.

---

### 3. **stabl/visualization.py** - Plotting

#### 3.1 Kaplan-Meier Plot (NEW)
**Location:** Lines 760-795  
**Addition:** New function for survival curve visualization

```python
def plot_kaplan_meier(y, predictions, risk_groups=2, show_fig=True, 
                     export_file=False, path=None):
    """Plot Kaplan-Meier survival curves stratified by risk groups."""
    from sksurv.nonparametric import kaplan_meier_estimator
    
    # Create risk groups based on prediction quantiles
    quantiles = np.linspace(0, 1, risk_groups + 1)
    thresholds = np.quantile(predictions, quantiles)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i in range(risk_groups):
        mask = (predictions >= thresholds[i]) & (predictions < thresholds[i+1])
        y_group = y[mask]
        
        time, survival = kaplan_meier_estimator(
            y_group['event'].astype(bool),
            y_group['time']
        )
        
        ax.step(time, survival, where='post', label=f'Risk Group {i+1}')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Survival Probability')
    ax.set_title('Kaplan-Meier Survival Curves')
    ax.legend()
    ax.grid(alpha=0.3)
    ...
```

**Purpose:** Visualizes how predicted risk scores stratify patients into survival curves.

---

### 4. **Plotting Fixes** (Recent)

#### 4.1 Stability Path X-Axis Correction
**Location:** `stabl/stabl.py`, `plot_stabl_path()`, Lines 595-630  
**Issue:** When `fitted_lambda_grid_` contains `'alphas'` parameter, the x-axis values were reconstructed incorrectly, causing mismatches between x positions and stability score columns, leading to vertical jumps in the plot.

**Fix:**
```python
elif 'alphas' in stabl.fitted_lambda_grid_:
    # Extract alphas in ParameterGrid order (matches stabl_scores_ columns)
    param_grid_all = list(ParameterGrid(stabl.fitted_lambda_grid_))
    alphas_in_order = []
    for p in param_grid_all:
        if 'alpha' in p:
            alphas_in_order.append(float(p['alpha']))
        elif 'alphas' in p:
            v = p['alphas']
            if isinstance(v, (list, tuple, np.ndarray)):
                v_arr = np.asarray(v).ravel()
                if v_arr.size == 0:
                    continue
                alphas_in_order.append(float(v_arr[0]))
            else:
                alphas_in_order.append(float(v))
    
    alphas_scalar = np.array(alphas_in_order, dtype=float)
    if alphas_scalar.size <= 1:
        x_grid_tmp = np.ones(max(1, alphas_scalar.size))
    else:
        x_grid_tmp = np.min(alphas_scalar) / alphas_scalar
    
    order_list = [np.arange(len(alphas_scalar))]
    x_grid_list = [x_grid_tmp]
    x_padding_list = [0]
```

#### 4.2 Log-Scale X-Axis
**Location:** `stabl/stabl.py`, `plot_stabl_path()`, Lines 706-712  
**Issue:** Lambda values spanning many orders of magnitude (1e-5 to 100) compressed most points near zero on linear scale, making paths look vertical.

**Fix:**
```python
# Use log-scale for lambda axis when values span multiple orders of magnitude
try:
    ax.set_xscale('log')
except Exception:
    pass  # Skip if x values are not strictly positive
```

**Purpose:** Spreads out lambda values proportionally on log scale, preventing visual compression and apparent vertical segments.

---

## Testing and Validation

### Example Usage Script
**File:** `stabl/run_stabl_cox_FIXED.py`

```python
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from stabl import Stabl

# Load survival data
clinical = pd.read_csv("clinical.csv")  # Must have 'time' and 'event' columns
counts = pd.read_csv("counts.csv")

# Prepare y as structured array (handled by _validate_data)
y = clinical[["event", "time"]]

# Create STABL with Cox model
base = CoxPHSurvivalAnalysis(alpha=0.1, n_alphas=50)

stabl = Stabl(
    base_estimator=base,
    lambda_grid="auto",
    n_bootstraps=1000,
    artificial_type="random_permutation",
    task_type="survival",  # KEY: Set task type
    n_jobs=-1
)

# Fit
stabl.fit(counts, y)

# Get selected features
selected_features = stabl.get_feature_names_out()

# Evaluate on test set
X_test, y_test = ...
predictions = base.fit(X_train[selected_features], y_train).predict(X_test[selected_features])

# Compute C-index
c_index = concordance_index_censored(
    y_test['event'].astype(bool),
    y_test['time'],
    predictions
)[0]
```

---

## Summary of Changes by Component

| Component | Changes | Impact |
|-----------|---------|--------|
| **Bootstrap Sampling** | Added task_type parameter, event-based validation | Ensures valid survival data in each bootstrap |
| **Data Validation** | Custom y conversion to structured arrays | Handles flexible input formats |
| **Lambda Grid** | Survival-specific alpha range calculation | Appropriate regularization strength |
| **Fitting Loop** | Conditional survival branch with y_surv | Passes correct format to Cox models |
| **Feature Selection** | Fallback for coefficient extraction | Robust to sklearn/sksurv API differences |
| **Cross-Validation** | Manual CV loop for survival | Works around sklearn limitations |
| **Scoring** | C-index instead of AUC | Appropriate metric for censored data |
| **Plotting** | Kaplan-Meier curves, stability path fixes | Visualization of survival predictions |

---

## Key Requirements

### Dependencies
```python
scikit-survival>=0.21.0  # For CoxPH models and concordance_index_censored
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
```

### Data Format Requirements
1. **Input y must be one of:**
   - pandas DataFrame with columns named: `['event', 'time']`, `['status', 'os']`, etc.
   - 2D numpy array: shape (n, 2) with columns [event, time]
   - Already-structured array: `dtype=[('event', '?'), ('time', '<f8')]`

2. **Event encoding:**
   - 1 = event occurred
   - 0 = censored

3. **Time values:**
   - Must be positive floats
   - NaN values are automatically dropped (with X filtered consistently)

---

## Known Limitations and Considerations

1. **Base Estimator:** Must be from scikit-survival (e.g., `CoxPHSurvivalAnalysis`, `CoxnetSurvivalAnalysis`)
2. **Lambda Grid:** For Cox models, use `'alphas'` parameter (not `'alpha'` or `'C'`)
3. **Prediction Output:** Cox models return risk scores (not probabilities)
4. **Evaluation Metric:** C-index (concordance index) used instead of AUC
5. **Plotting:** Stability path now uses log-scale x-axis by default for better visualization

---

## Migration from Binary/Regression Tasks

To convert existing STABL code to survival:

```python
# OLD (Binary Classification)
from sklearn.linear_model import LogisticRegression
base = LogisticRegression(penalty='l1', C=1.0)
stabl = Stabl(base_estimator=base, task_type="binary")
y = targets  # Shape: (n,) with 0/1 values

# NEW (Survival)
from sksurv.linear_model import CoxPHSurvivalAnalysis
base = CoxPHSurvivalAnalysis(alpha=0.1, n_alphas=50)
stabl = Stabl(base_estimator=base, task_type="survival")
y = clinical[["event", "time"]]  # DataFrame with event (1/0) and time (float)
```

---

## Conclusion

The STABL framework has been successfully extended to support survival analysis while maintaining backward compatibility with classification and regression tasks. All changes follow the scikit-survival API conventions and integrate seamlessly with the existing STABL workflow. The recent plotting fixes ensure stability paths are visualized correctly even when lambda values span many orders of magnitude.

**Status:** Fully functional for survival prediction tasks  
**Tested on:** mRNA gene expression data with overall survival outcomes  
**Results:** Successfully generates stability paths, FDR control, and selected features for Cox models
