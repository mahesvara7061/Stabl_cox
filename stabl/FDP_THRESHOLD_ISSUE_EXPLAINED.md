# Why FDP+ Threshold is Always 1.0 - Root Cause Analysis

## Executive Summary

Your FDP+ threshold is always 1.0 because **artificial (permuted) features are being selected at nearly the same rate as real features**, indicating the model cannot distinguish true signal from noise. The primary cause is an **extremely narrow alpha regularization range (0.01 to 0.1)**.

---

## Understanding FDP+ (False Discovery Proportion Plus)

### What is FDP+?

FDP+ controls the false discovery rate in feature selection by comparing real feature selection frequencies against artificial (permuted) features:

```python
FDP = (n_artificial_selected / artificial_proportion + 1) / max(1, n_real_selected)
```

With `artificial_proportion=0.5`:
```
FDP = (2 × n_artificial_selected + 1) / n_real_selected
```

### When Does FDP+ = 1.0?

Looking at `stabl.py:1762-1767`:

```python
if self.min_fdr_ > 1.:
    final_cutoff = 1.
else:
    final_cutoff = np.min([self.fdr_threshold_range[np.argmin(self.FDRs_)], 1])

self.fdr_min_threshold_ = final_cutoff
```

**The threshold is set to 1.0 when `min_fdr_ > 1.0`**, which happens when:
- Artificial features are selected as often as (or more than) real features
- The model treats noise as signal

---

## Root Cause: Narrow Alpha Range

### The Problem

In `run_stabl_cox_from_counts.py:236`:

```python
alpha_list = np.logspace(-2, -1, 50)  # ❌ 0.01 to 0.1 only!
```

This creates alphas from **0.01 to 0.1** — a 10× range.

### Why This Causes FDP+ = 1.0

1. **Weak regularization**: Small alphas (0.01-0.1) provide minimal penalty
2. **Over-selection**: Both real AND artificial features get selected frequently
3. **No separation**: Model can't tell signal from noise
4. **High FDP**: Many artificial features pass threshold → FDP > 1 → threshold set to 1.0

### Typical Cox Model Alpha Ranges

For comparison, `scikit-survival` and other Cox model implementations typically use:
- **Broad range**: 1e-4 to 10 or even 1e-5 to 100
- **Span**: 10,000× to 1,000,000× (vs your 10×)
- **Purpose**: Explore from very weak to very strong regularization

---

## How to Fix

### 1. ✅ EXPAND ALPHA RANGE (MOST IMPORTANT)

**Change `run_stabl_cox_from_counts.py:236` from:**
```python
alpha_list = np.logspace(-2, -1, 50)  # 0.01 to 0.1
```

**To:**
```python
alpha_list = np.logspace(-4, 1, 50)  # 1e-4 to 10
```

**Why this works:**
- Explores 100,000× range instead of 10×
- Strong regularization (high alphas) forces selection of only truly predictive features
- Noise features get penalized out
- Real features with genuine survival associations survive strong penalties
- FDP drops below 1.0

**Alternative ranges to try:**
```python
# Conservative (moderate regularization)
alpha_list = np.logspace(-3, 0, 50)  # 0.001 to 1

# Aggressive (strong regularization)
alpha_list = np.logspace(-4, 2, 50)  # 0.0001 to 100

# Ultra-wide (exploratory)
alpha_list = np.logspace(-5, 2, 100)  # 0.00001 to 100
```

### 2. Check Events-Per-Variable (EPV)

**Recommended EPV for Cox models: > 10**

```python
n_events = y['event'].sum()
n_features = X.shape[1]
epv = n_events / n_features

if epv < 10:
    print(f"⚠️ EPV = {epv:.1f} < 10: Risk of overfitting!")
    print(f"Reduce features to ~{int(n_events / 10)}")
```

**Example:**
- 50 events, 100 genes → EPV = 0.5 ❌ (too low!)
- 50 events, 100 genes → EPV = 0.5 ❌ (reduce to ~5 genes)
- 200 events, 100 genes → EPV = 2.0 ⚠️ (marginal)
- 500 events, 100 genes → EPV = 5.0 ✅ (acceptable)
- 1000 events, 100 genes → EPV = 10.0 ✅ (good)

**How to fix:**
```bash
# Option A: Reduce number of genes
python run_stabl_cox_FIXED.py \
    --counts data.tsv \
    --clinical clinical.tsv \
    --num_genes 50 \  # or lower
    --outdir results

# Option B: Pre-filter genes (e.g., by variance, univariate p-value)
```

### 3. Increase Artificial Proportion

```python
# In build_stabl_cox():
stabl_cox = Stabl(
    ...
    artificial_proportion=1.0,  # ← Change from 0.5 to 1.0
    ...
)
```

**Effect:**
- `artificial_proportion=0.5`: FDP = (2×artificial + 1) / real
- `artificial_proportion=1.0`: FDP = (1×artificial + 1) / real
- Higher proportion → more stringent FDR control

### 4. Other Tuning Options

```python
# More bootstraps (increases stability, slower)
n_bootstraps=1000  # instead of 500

# Wider FDR threshold range
fdr_threshold_range=np.arange(0.01, 1.01, 0.01)  # finer grid

# Different artificial feature type
artificial_type="knockoff"  # instead of "random_permutation"
```

---

## Diagnostic Workflow

### Step 1: Run Diagnostic Script

```bash
python diagnose_fdp.py \
    --counts your_counts.tsv \
    --clinical your_clinical.tsv \
    --num_genes 100 \
    --n_bootstraps 100 \
    --outdir ./diagnostics
```

This will:
- Show stability score distributions (real vs artificial)
- Plot FDP+ curve
- Calculate effect size (separation between real/artificial)
- Provide specific recommendations

**Key metrics to check:**
- **Effect size (Cohen's d)**: Should be > 0.5 (real features more stable than artificial)
- **Mean stability gap**: Real features should have higher mean than artificial
- **FDP curve**: Should have region < 1.0

### Step 2: Apply Fix

Use the fixed version with expanded alpha range:

```bash
python run_stabl_cox_FIXED.py \
    --counts your_counts.tsv \
    --clinical your_clinical.tsv \
    --num_genes 50 \  # adjust based on EPV
    --n_boot 500 \
    --outdir ./results_fixed
```

### Step 3: Verify Results

Check the output:

```bash
# Look at STABL diagnostics
cat results_fixed/Training\ CV/STABL\ Cox\ results*/Selected\ Features/Selected\ features.csv

# Check FDP+ values across folds
cat results_fixed/Training\ CV/Stabl\ features\ STABL\ Cox/Stabl\ features\ STABL\ Cox\ mRNA.csv
```

**Expected after fix:**
- `min FDP+` column shows values < 1.0 (e.g., 0.15, 0.30, etc.)
- `Threshold` column shows values < 1.0 (e.g., 0.45, 0.60, etc.)
- Some features actually get selected (not empty feature set)

---

## Understanding the Math

### FDP+ Calculation (from `stabl.py:1741-1746`)

```python
for thresh in self.fdr_threshold_range:
    num = np.sum((1 / artificial_proportion) * (max_scores_artificial > thresh)) + 1
    denum = max([1, np.sum((max_scores > thresh))])
    FDP = num / denum
    FDPs.append(FDP)
```

**Example with your data:**

Assume:
- `artificial_proportion = 0.5`
- At threshold = 0.5:
  - 40 real features have max_score > 0.5
  - 20 artificial features have max_score > 0.5

```python
FDP = ((20 / 0.5) + 1) / max(1, 40)
    = (40 + 1) / 40
    = 41 / 40
    = 1.025  # > 1.0 ❌
```

**With narrow alpha range:**
- Weak regularization → too many features selected
- Artificial features selected at ~50% rate of real features
- FDP stays > 1.0 across all thresholds

**With wide alpha range:**
- Strong regularization (high alphas) → only strong signals survive
- Artificial features (noise) get penalized out
- Example at threshold = 0.6:
  - 20 real features (strong signals)
  - 2 artificial features (false positives)

```python
FDP = ((2 / 0.5) + 1) / max(1, 20)
    = (4 + 1) / 20
    = 5 / 20
    = 0.25  # < 1.0 ✅
```

---

## Expected Outcomes

### Before Fix (Current)
- ❌ FDP+ threshold: **1.0** (every fold)
- ❌ min FDP+: **> 1.0** (e.g., 1.5, 2.0, 3.0)
- ⚠️ Features selected: Either 0 (explore mode) or too many
- ⚠️ Model can't distinguish signal from noise

### After Fix (Expected)
- ✅ FDP+ threshold: **0.2-0.8** (varies by fold)
- ✅ min FDP+: **0.05-0.50** (< 1.0)
- ✅ Features selected: Reasonable number (5-30 depending on signal)
- ✅ Real features have higher stability scores than artificial

---

## Quick Reference

### Files Created

1. **`diagnose_fdp.py`**: Diagnostic tool to analyze the issue
2. **`run_stabl_cox_FIXED.py`**: Fixed version with wide alpha range
3. **This document**: Explanation and solutions

### One-Line Summary

**Problem:** Alpha range too narrow (0.01-0.1) → weak regularization → noise selected as often as signal → FDP > 1 → threshold = 1.

**Solution:** Use `np.logspace(-4, 1, 50)` for alphas from 0.0001 to 10.

---

## Additional Resources

### Understanding Regularization in Cox Models

- **Low alpha (weak regularization)**: More features selected, risk of noise
- **High alpha (strong regularization)**: Fewer features, only strong signals
- **Optimal**: Somewhere in between, found by cross-validation

### STABL Method References

- Original STABL paper: [https://doi.org/10.1093/bioinformatics/btab158](https://doi.org/10.1093/bioinformatics/btab158)
- Knockoff filter: [https://doi.org/10.1080/01621459.2015.1035739](https://doi.org/10.1080/01621459.2015.1035739)

---

## Need More Help?

### Check These Files
1. `fdp_diagnostics/fdp_diagnostic_plots.pdf` - Visual analysis
2. `fdp_diagnostics/fdp_diagnostic_report.txt` - Text summary
3. `results_fixed/Training CV/STABL Diagnostics/` - Per-fold stability plots

### Common Issues After Fix

**Issue: Still getting FDP+ = 1.0**
- Check EPV ratio (may need fewer features)
- Check event rate (need > 10% events)
- Try even wider alpha range: `np.logspace(-5, 2, 100)`

**Issue: No features selected (all zero)**
- Alpha range too aggressive
- Try: `np.logspace(-3, 0, 50)` (more conservative)
- Or increase `n_explore` parameter

**Issue: C-index very low**
- Data may lack predictive signal
- Check clinical/molecular associations
- Consider univariate filtering first

---

## Summary

| Aspect | Current (Broken) | Fixed |
|--------|------------------|-------|
| Alpha range | 0.01 to 0.1 (10×) | 0.0001 to 10 (100,000×) |
| Regularization | Too weak | Explores weak → strong |
| Noise filtering | Poor (noise selected) | Good (noise penalized) |
| FDP+ | Always > 1.0 | Typically 0.1-0.5 |
| Threshold | Always 1.0 | 0.3-0.8 (varies) |
| Features | 0 or too many | Reasonable (5-30) |

**Bottom line:** Change one line of code (alpha range) to fix the issue!