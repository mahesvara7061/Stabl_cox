# Alternative Regularized Survival Models for STABL

## Problem Statement

`CoxnetSurvivalAnalysis` is sensitive to alpha values and produces warnings:
- "alpha too small" when alpha < 1e-3
- "alpha too large" when alpha > 5
- Makes STABL feature selection process noisy with warnings

## Available Options in scikit-survival

### 1. ✅ **FastCoxPH** (Recommended)
```python
from sksurv.linear_model import CoxPHSurvivalAnalysis

model = CoxPHSurvivalAnalysis(
    alpha=0.1,        # Ridge penalty (L2)
    ties='breslow',
    n_iter=1000
)
```

**Pros**:
- Simple L2 (Ridge) regularization
- Single alpha parameter (not array)
- Very stable across wide range of alphas
- Fast convergence
- No warnings

**Cons**:
- Only Ridge (no LASSO, no sparsity)
- Less feature selection within each bootstrap

**Use Case**: When you want stable, reliable regularization without sparsity

---

### 2. ⚠️ **CoxnetSurvivalAnalysis** (Current, Problematic)
```python
from sksurv.linear_model import CoxnetSurvivalAnalysis

model = CoxnetSurvivalAnalysis(
    l1_ratio=0.9,
    alphas=np.array([0.1]),
    max_iter=100000
)
```

**Pros**:
- Elastic net (L1 + L2)
- Sparse solutions
- Handles correlated features

**Cons**:
- ❌ Very sensitive to alpha values
- ❌ Produces warnings with extreme alphas
- ❌ Requires array input for alphas
- ❌ Can fail to converge

**Use Case**: When you need sparsity AND can control alpha range tightly

---

### 3. ✅ **FastSurvivalSVM** (Alternative)
```python
from sksurv.svm import FastSurvivalSVM

model = FastSurvivalSVM(
    alpha=0.1,        # Regularization strength
    rank_ratio=1.0,
    fit_intercept=False,
    max_iter=1000
)
```

**Pros**:
- Different approach (ranking-based)
- Stable across alpha range
- Good for non-linear relationships
- No warnings

**Cons**:
- Different prediction scale (harder to interpret)
- Slower than Cox
- Less familiar to biomedical researchers

**Use Case**: When Cox assumptions may be violated

---

### 4. ❌ **ComponentwiseGradientBoostingSurvivalAnalysis**
```python
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis

model = ComponentwiseGradientBoostingSurvivalAnalysis(
    n_estimators=100,
    learning_rate=0.1
)
```

**Pros**:
- Built-in feature selection
- Non-linear relationships

**Cons**:
- ❌ No direct regularization parameter to grid search
- ❌ Too complex for STABL's grid search framework
- ❌ Slower

**Use Case**: Not suitable for STABL

---

## Recommendation

### Option A: Switch to Ridge-Penalized Cox (CoxPHSurvivalAnalysis) ✅

**Why**:
- Very stable, no warnings
- Wide alpha range works: 1e-5 to 100
- Fast and reliable
- Still regularized (just L2 instead of elastic net)

**Trade-off**:
- No sparsity within each bootstrap
- But STABL provides sparsity through stability selection!

**Implementation**:
```python
def build_stabl_cox(n_bootstraps=500, random_state=42):
    alpha_list = np.logspace(-5, 2, 50)  # 1e-5 to 100, very wide range
    
    lambda_grid = {
        "alpha": alpha_list  # Note: NOT in nested arrays
    }
    
    base = CoxPHSurvivalAnalysis(
        ties='breslow',
        n_iter=1000,
        tol=1e-7,
        verbose=0
    )
    
    stabl_cox = Stabl(
        base_estimator=base,
        lambda_grid=lambda_grid,
        n_bootstraps=n_bootstraps,
        task_type="survival",
        ...
    )
    return stabl_cox
```

**Final model** (after feature selection):
```python
# Still use CoxnetSurvivalAnalysis for final model (with known good alpha)
final_cox = CoxnetSurvivalAnalysis(
    l1_ratio=0.9,
    alphas=np.array([0.1]),  # Safe value
    max_iter=100000,
    tol=1e-7
).fit(X_selected, y)
```

---

### Option B: Restrict Coxnet Alpha Range ⚠️

**Why**:
- Keep elastic net benefits
- Just avoid problematic alphas

**Implementation**:
```python
def build_stabl_cox(n_bootstraps=500, random_state=42):
    # Restricted range: avoid extremes
    alpha_list = np.logspace(-3, 0.5, 50)  # 0.001 to 3.16
    
    lambda_grid = {
        "alphas": [np.array([a]) for a in alpha_list]
    }
    
    base = CoxnetSurvivalAnalysis(
        l1_ratio=0.7,
        tol=1e-6,
        max_iter=2_000_000,
        verbose=0  # Suppress warnings
    )
    
    stabl_cox = Stabl(
        base_estimator=base,
        lambda_grid=lambda_grid,
        ...
    )
    return stabl_cox
```

**Trade-off**:
- Narrower regularization exploration
- May not separate signal from noise as well
- Still get some warnings at edges

---

## Comparison Table

| Model | Regularization | Stability | Warnings | Speed | Interpretability |
|-------|---------------|-----------|----------|-------|------------------|
| **CoxPH (Ridge)** | ✅ L2 only | ✅✅✅ Excellent | ✅ None | ✅✅ Fast | ✅✅ Simple |
| **Coxnet (Full Range)** | ✅✅ L1+L2 | ⚠️ Sensitive | ❌ Many | ⚠️ Slower | ✅✅ Simple |
| **Coxnet (Restricted)** | ✅✅ L1+L2 | ⚠️ Moderate | ⚠️ Some | ⚠️ Slower | ✅✅ Simple |
| **SurvivalSVM** | ✅ L2 only | ✅✅ Good | ✅ None | ❌ Slow | ⚠️ Different |

---

## My Recommendation: **Option A (Ridge Cox)** ✅

### Rationale:

1. **STABL already provides sparsity**
   - Through stability selection and FDR control
   - Don't need L1 penalty in base estimator

2. **Ridge is sufficient**
   - Handles collinearity (main concern)
   - Stabilizes coefficients
   - No sparsity needed per-bootstrap

3. **Much more stable**
   - No alpha warnings
   - Wider range exploration
   - Better signal/noise separation

4. **Final model can still use elastic net**
   - After features selected, use CoxnetSurvivalAnalysis
   - Controlled alpha value (0.1)
   - Best of both worlds

### Two-Stage Approach:

**Stage 1: Feature Selection (STABL)**
- Use `CoxPHSurvivalAnalysis` (Ridge)
- Wide alpha range: 1e-5 to 100
- Stable, no warnings
- Selects stable features

**Stage 2: Final Model**
- Use `CoxnetSurvivalAnalysis` (Elastic Net)
- Fixed alpha: 0.1
- On selected features only
- Sparse final coefficients

---

## Implementation Plan

### Files to Modify:

1. **`run_stabl_cox_FIXED.py`**
   - Change base estimator to CoxPHSurvivalAnalysis
   - Update lambda_grid format
   - Update documentation

2. **`multi_omic_pipelines.py`**
   - Update validation check (line 221)
   - Keep final model as CoxnetSurvivalAnalysis (lines 426, 654)

3. **Documentation**
   - Update to explain two-stage approach
   - Clarify why Ridge for selection, Elastic Net for final model

---

## Code Changes Preview

### Current (v3):
```python
base = CoxnetSurvivalAnalysis(
    l1_ratio=0.7,
    tol=1e-6,
    max_iter=2_000_000
)

lambda_grid = {
    "alphas": [np.array([a]) for a in alpha_list]  # Nested arrays
}
```

### Proposed (v4):
```python
base = CoxPHSurvivalAnalysis(
    ties='breslow',
    n_iter=1000,
    tol=1e-7,
    verbose=0
)

lambda_grid = {
    "alpha": alpha_list  # Simple array
}
```

---

## Expected Improvements

### Before (v3):
```
⚠️ Warning: alpha too small, using minimum alpha
⚠️ Warning: alpha too large, using maximum alpha
⚠️ Warning: convergence not reached
...
[100 warnings during STABL]
```

### After (v4):
```
[INFO] Alpha range: 0.00001 to 100.00
[INFO] Starting bootstrap 1/500...
[INFO] Starting bootstrap 2/500...
...
[No warnings]
```

---

## Bottom Line

**Switch base estimator from CoxnetSurvivalAnalysis to CoxPHSurvivalAnalysis**

- ✅ Eliminates all alpha warnings
- ✅ More stable feature selection
- ✅ Wider regularization exploration
- ✅ Faster execution
- ✅ Still use elastic net for final model
- ✅ Better overall results

**This is the right approach because STABL's stability selection provides the sparsity, so we don't need L1 penalty in every bootstrap iteration.**
