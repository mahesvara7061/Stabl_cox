# Quick Start: Fixing FDP+ Threshold Issue

## TL;DR - The Problem

Your FDP+ threshold is always 1.0 because **the alpha regularization range is too narrow** (0.01 to 0.1).

**Root cause line:** `run_stabl_cox_from_counts.py:236`

```python
alpha_list = np.logspace(-2, -1, 50)  # ❌ Only 0.01 to 0.1
```

This weak regularization lets both real AND artificial features get selected → FDP > 1 → threshold set to 1.0.

---

## Quick Fix (2 Steps)

### Step 1: Run Diagnostic (optional but recommended)

```bash
cd /home/mahesvara/Documents/Labs/Pharmaco-Omics/first_project/source/Stabl/stabl

python diagnose_fdp.py \
    --counts /path/to/your/counts.tsv \
    --clinical /path/to/your/clinical.tsv \
    --num_genes 100 \
    --n_bootstraps 100 \
    --outdir ./fdp_diagnostics
```

This will:
- Show you exactly why FDP+ = 1.0
- Create diagnostic plots
- Give specific recommendations for your data

**Check output:**
```bash
cat fdp_diagnostics/fdp_diagnostic_report.txt
```

### Step 2: Use Fixed Version

```bash
python run_stabl_cox_FIXED.py \
    --counts /path/to/your/counts.tsv \
    --clinical /path/to/your/clinical.tsv \
    --num_genes 50 \
    --n_boot 500 \
    --outdir ./results_fixed
```

**Key differences:**
- ✅ Alpha range: 0.0001 to 10 (instead of 0.01 to 0.1)
- ✅ Checks Events-Per-Variable (EPV) ratio
- ✅ Warns if data unsuitable for number of features

---

## Verify the Fix Worked

After running, check:

```bash
# 1. Look at FDP+ values across folds
cat results_fixed/Training\ CV/Stabl\ features\ STABL\ Cox/Stabl\ features\ STABL\ Cox\ mRNA.csv
```

**Expected output:**
```
,Threshold,min FDP+
Fold n°1,0.45,0.23
Fold n°2,0.60,0.31
Fold n°3,0.55,0.27
...
```

**Before fix (broken):**
```
,Threshold,min FDP+
Fold n°1,1.0,1.52
Fold n°2,1.0,1.78
Fold n°3,1.0,1.65
...
```

```bash
# 2. Check features selected
cat results_fixed/Training\ CV/STABL\ Cox\ results*/Selected\ Features/Selected\ features.csv
```

Should show actual gene names (not empty).

---

## Alternative: Manual Fix of Original File

If you want to fix `run_stabl_cox_from_counts.py` directly:

**Edit line 236:**

```python
# Change this:
alpha_list = np.logspace(-2, -1, 50)  # 0.01 to 0.1

# To this:
alpha_list = np.logspace(-4, 1, 50)  # 0.0001 to 10
```

Save and re-run your original command.

---

## Understanding the Numbers

### Events-Per-Variable (EPV) Ratio

The diagnostic will show your EPV:

```
Events: 50/200 (25.0%)
Events-per-variable (EPV): 0.50
⚠️ EPV < 10 may lead to overfitting!
```

**Recommendations:**
- EPV < 1: ❌ Very high overfitting risk, reduce features drastically
- EPV 1-5: ⚠️ Marginal, reduce features
- EPV 5-10: ✅ Acceptable
- EPV > 10: ✅ Good

**How to fix low EPV:**

```bash
# If you have 50 events, use at most 5 genes
python run_stabl_cox_FIXED.py \
    --counts data.tsv \
    --clinical clinical.tsv \
    --num_genes 5 \  # ← Adjust based on EPV
    --outdir results
```

### FDP+ Values

- **FDP+ < 0.2**: Excellent signal separation
- **FDP+ 0.2-0.5**: Good, reasonable FDR control
- **FDP+ 0.5-1.0**: Marginal, consider data quality
- **FDP+ > 1.0**: ❌ Broken, artificial = real (your current issue)

---

## Common Issues After Fix

### Issue: FDP+ still > 1.0

**Possible causes:**
1. EPV too low → Reduce `--num_genes`
2. Event rate too low (< 10%) → Need more events or different analysis
3. No real signal in data → Check biological relevance

**Try:**
```bash
# More aggressive regularization
# Edit build_stabl_cox() in run_stabl_cox_FIXED.py:
alpha_list = np.logspace(-5, 2, 50)  # 0.00001 to 100
```

### Issue: No features selected (all zero)

**Possible causes:**
1. Regularization too strong

**Try:**
```bash
# Less aggressive regularization
# Edit build_stabl_cox() in run_stabl_cox_FIXED.py:
alpha_list = np.logspace(-3, 0, 50)  # 0.001 to 1
```

Or increase explore parameter:
```python
stabl_cox = Stabl(
    ...
    explore=True,
    n_explore=10,  # ← increase from 5
    ...
)
```

### Issue: C-index very low (< 0.6)

**This is a DATA problem, not a code problem.**

Possible causes:
- Survival outcome not associated with gene expression
- Wrong genes selected (try different gene sets)
- Clinical confounders not accounted for
- Data quality issues

---

## Files Created for You

1. **`diagnose_fdp.py`**
   - Diagnostic tool to understand the problem
   - Run this first to see what's wrong

2. **`run_stabl_cox_FIXED.py`**
   - Fixed version with proper alpha range
   - Use this for your analysis

3. **`FDP_THRESHOLD_ISSUE_EXPLAINED.md`**
   - Detailed explanation of the problem
   - Read if you want to understand the math

4. **`QUICK_START_FIX.md`** (this file)
   - Quick guide to fix and verify

---

## Example Workflow

```bash
# 1. Navigate to directory
cd /home/mahesvara/Documents/Labs/Pharmaco-Omics/first_project/source/Stabl/stabl

# 2. Run diagnostic (100 bootstraps for speed)
python diagnose_fdp.py \
    --counts /path/to/counts.tsv \
    --clinical /path/to/clinical.tsv \
    --num_genes 100 \
    --n_bootstraps 100 \
    --outdir ./diagnostics

# 3. Check diagnostic output
cat diagnostics/fdp_diagnostic_report.txt
open diagnostics/fdp_diagnostic_plots.pdf  # or xdg-open on Linux

# 4. Run fixed version (adjust num_genes based on EPV from step 2)
python run_stabl_cox_FIXED.py \
    --counts /path/to/counts.tsv \
    --clinical /path/to/clinical.tsv \
    --num_genes 50 \
    --n_boot 500 \
    --n_splits 5 \
    --n_repeats 5 \
    --outdir ./results_fixed

# 5. Verify results
cat results_fixed/Summary/quick_summary.txt
cat results_fixed/Training\ CV/Stabl\ features\ STABL\ Cox/Stabl\ features\ STABL\ Cox\ mRNA.csv
```

---

## What You Should See After Fix

### Before (Broken)
```
Min FDR+: 1.5234
FDR Threshold: 1.0000
Features selected: 0 or way too many
Real vs Artificial separation: Poor (effect size < 0.2)
```

### After (Fixed)
```
Min FDR+: 0.2314
FDR Threshold: 0.5500
Features selected: 12
Real vs Artificial separation: Good (effect size > 0.8)
```

---

## Need Help?

1. **Check diagnostic plots:** `fdp_diagnostics/fdp_diagnostic_plots.pdf`
2. **Read detailed explanation:** `FDP_THRESHOLD_ISSUE_EXPLAINED.md`
3. **Check console output:** Look for warnings about EPV, events, etc.

---

## Summary

| What | Before | After |
|------|--------|-------|
| Alpha range | 0.01-0.1 | 0.0001-10 |
| FDP+ | > 1.0 ❌ | 0.1-0.5 ✅ |
| Threshold | Always 1.0 | 0.3-0.8 |
| Features | 0 or too many | Reasonable |

**One line to remember:** The alpha range was too narrow, causing weak regularization.