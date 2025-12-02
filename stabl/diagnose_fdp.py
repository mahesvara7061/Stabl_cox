#!/usr/bin/env python3
"""
Diagnostic script to understand why FDP+ threshold is always 1.
This script analyzes the stability scores of real vs artificial features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys

# Import from your stabl module
from run_stabl_cox_from_counts import load_counts, load_clinical, align_X_y, build_stabl_cox


def diagnose_fdp_issue(counts_path, clinical_path, num_genes=100, n_bootstraps=100, output_dir="./fdp_diagnostics"):
    """
    Run STABL Cox and diagnose why FDP+ is high.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("="*70)
    print("FDP+ DIAGNOSTIC ANALYSIS")
    print("="*70)

    # Load data
    print("\n1. Loading data...")
    X = load_counts(counts_path, num_genes=num_genes)
    y = load_clinical(clinical_path)
    X, y = align_X_y(X, y)

    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_events = y['event'].sum()
    n_censored = (~y['event']).sum()

    print(f"   Samples: {n_samples}")
    print(f"   Features: {n_features}")
    print(f"   Events: {n_events} ({100*n_events/n_samples:.1f}%)")
    print(f"   Censored: {n_censored} ({100*n_censored/n_samples:.1f}%)")
    print(f"   Events-per-variable (EPV): {n_events / n_features:.2f}")

    if n_events / n_features < 10:
        print(f"   ⚠️  WARNING: EPV < 10 indicates potential overfitting risk!")

    # Build STABL with fewer bootstraps for diagnosis
    print(f"\n2. Fitting STABL Cox (n_bootstraps={n_bootstraps})...")
    stabl = build_stabl_cox(n_bootstraps=n_bootstraps, random_state=42)

    # Show the alpha grid
    print(f"\n3. Alpha grid analysis:")
    alpha_array = [float(a[0]) for a in stabl.lambda_grid["alphas"]]
    print(f"   Min alpha: {min(alpha_array):.6f}")
    print(f"   Max alpha: {max(alpha_array):.6f}")
    print(f"   Number of alphas: {len(alpha_array)}")
    print(f"   Range span: {max(alpha_array) / min(alpha_array):.1f}x")

    if max(alpha_array) / min(alpha_array) < 100:
        print(f"   ⚠️  WARNING: Alpha range is very narrow! Consider 1e-4 to 1e1.")

    # Fit the model
    stabl.fit(X, y)

    # Analyze results
    print(f"\n4. STABL Results:")
    print(f"   Min FDR+: {stabl.min_fdr_:.4f}")
    print(f"   FDR threshold: {stabl.fdr_min_threshold_:.4f}")

    if stabl.min_fdr_ > 1.0:
        print(f"   ❌ PROBLEM: min_fdr_ > 1 → threshold set to 1.0")
        print(f"   This means artificial features are selected as often as real ones!")

    # Analyze stability scores
    max_scores_real = np.max(stabl.stabl_scores_, axis=1)
    max_scores_artificial = np.max(stabl.stabl_scores_artificial_, axis=1)

    print(f"\n5. Stability Score Statistics:")
    print(f"   Real features:")
    print(f"     Mean: {np.mean(max_scores_real):.4f}")
    print(f"     Median: {np.median(max_scores_real):.4f}")
    print(f"     Max: {np.max(max_scores_real):.4f}")
    print(f"     Std: {np.std(max_scores_real):.4f}")

    print(f"   Artificial features:")
    print(f"     Mean: {np.mean(max_scores_artificial):.4f}")
    print(f"     Median: {np.median(max_scores_artificial):.4f}")
    print(f"     Max: {np.max(max_scores_artificial):.4f}")
    print(f"     Std: {np.std(max_scores_artificial):.4f}")

    # Separation analysis
    mean_diff = np.mean(max_scores_real) - np.mean(max_scores_artificial)
    effect_size = mean_diff / np.sqrt((np.std(max_scores_real)**2 + np.std(max_scores_artificial)**2) / 2)

    print(f"\n6. Separation Analysis:")
    print(f"   Mean difference: {mean_diff:.4f}")
    print(f"   Effect size (Cohen's d): {effect_size:.4f}")

    if effect_size < 0.5:
        print(f"   ❌ PROBLEM: Poor separation (d < 0.5)!")
        print(f"      Real features aren't much better than noise.")

    # FDP calculation at different thresholds
    print(f"\n7. FDP+ at Different Thresholds:")
    test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    artificial_prop = stabl.artificial_proportion

    print(f"   Threshold | Real>T | Artif>T | FDP+")
    print(f"   {'-'*45}")
    for thresh in test_thresholds:
        n_real = np.sum(max_scores_real > thresh)
        n_artif = np.sum(max_scores_artificial > thresh)
        fdp = ((n_artif / artificial_prop) + 1) / max(1, n_real)
        print(f"   {thresh:5.2f}     | {n_real:6d} | {n_artif:7d} | {fdp:.4f}")

    # Visualizations
    print(f"\n8. Creating diagnostic plots...")

    # Plot 1: Distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Histogram
    ax = axes[0, 0]
    bins = np.linspace(0, 1, 50)
    ax.hist(max_scores_real, bins=bins, alpha=0.6, label='Real features', color='blue', density=True)
    ax.hist(max_scores_artificial, bins=bins, alpha=0.6, label='Artificial features', color='red', density=True)
    ax.axvline(stabl.fdr_min_threshold_, color='black', linestyle='--', linewidth=2, label=f'Threshold ({stabl.fdr_min_threshold_:.3f})')
    ax.set_xlabel('Max Stability Score')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Max Stability Scores')
    ax.legend()
    ax.grid(alpha=0.3)

    # Cumulative distribution
    ax = axes[0, 1]
    sorted_real = np.sort(max_scores_real)
    sorted_artif = np.sort(max_scores_artificial)
    ax.plot(sorted_real, np.linspace(0, 1, len(sorted_real)), label='Real features', linewidth=2)
    ax.plot(sorted_artif, np.linspace(0, 1, len(sorted_artif)), label='Artificial features', linewidth=2)
    ax.axvline(stabl.fdr_min_threshold_, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Max Stability Score')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution Functions')
    ax.legend()
    ax.grid(alpha=0.3)

    # FDP curve
    ax = axes[1, 0]
    ax.plot(stabl.fdr_threshold_range, stabl.FDRs_, linewidth=2, color='navy')
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='FDP=1')
    ax.axvline(stabl.fdr_min_threshold_, color='green', linestyle='--', linewidth=2, label=f'Min threshold ({stabl.fdr_min_threshold_:.3f})')
    ax.set_xlabel('Stability Threshold')
    ax.set_ylabel('FDP+')
    ax.set_title(f'FDP+ Curve (min={stabl.min_fdr_:.3f})')
    ax.legend()
    ax.grid(alpha=0.3)

    # Top features stability paths
    ax = axes[1, 1]
    n_top = min(20, len(max_scores_real))
    top_idx = np.argsort(max_scores_real)[-n_top:]
    for idx in top_idx:
        ax.plot(stabl.stabl_scores_[idx, :], alpha=0.5, linewidth=1.5, color='blue')
    # Plot some artificial features too
    n_artif_plot = min(10, len(max_scores_artificial))
    for idx in range(n_artif_plot):
        ax.plot(stabl.stabl_scores_artificial_[idx, :], alpha=0.3, linewidth=1, color='red', linestyle=':')
    ax.axhline(stabl.fdr_min_threshold_, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Lambda Index')
    ax.set_ylabel('Stability Score')
    ax.set_title(f'Stability Paths (Top {n_top} real + {n_artif_plot} artificial)')
    ax.legend(['Real features', 'Artificial features', 'Threshold'])
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "fdp_diagnostic_plots.pdf"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {plot_path}")

    # Summary recommendations
    print(f"\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("="*70)

    if stabl.min_fdr_ > 1.0:
        print("\n🔧 Your FDP+ threshold is 1.0 because artificial features are")
        print("   selected as frequently as real features. Try these fixes:\n")

        print("1. EXPAND ALPHA RANGE (MOST IMPORTANT):")
        print("   Current: 0.01 to 0.1 (too narrow)")
        print("   Try: np.logspace(-4, 1, 50)  # 1e-4 to 10")
        print("   This gives stronger regularization to separate signal from noise.\n")

        print("2. REDUCE NUMBER OF FEATURES:")
        print(f"   Current: {n_features} features with {n_events} events (EPV={n_events/n_features:.1f})")
        print(f"   Recommended EPV: > 10")
        print(f"   Try: --num_genes {max(10, int(n_events / 10))} or use pre-filtering.\n")

        print("3. INCREASE ARTIFICIAL PROPORTION:")
        print(f"   Current: artificial_proportion={artificial_prop}")
        print("   Try: artificial_proportion=1.0 (more stringent FDR control)\n")

        print("4. CHECK DATA QUALITY:")
        print("   - Ensure survival times are well-distributed")
        print("   - Check for outliers/data quality issues")
        print("   - Consider log-transforming heavily skewed survival times\n")

        print("5. INCREASE BOOTSTRAPS:")
        print(f"   Current: {n_bootstraps} (for this diagnostic)")
        print("   Recommended: 500-1000 for final analysis\n")

    else:
        print(f"\n✅ FDP+ control is working (min_fdr_={stabl.min_fdr_:.3f} < 1.0)")
        print(f"   Threshold: {stabl.fdr_min_threshold_:.3f}")
        print(f"   Features selected: {stabl.get_support().sum()}")

    print("\n" + "="*70)

    # Save detailed report
    report_path = output_dir / "fdp_diagnostic_report.txt"
    with open(report_path, 'w') as f:
        f.write("FDP+ DIAGNOSTIC REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Data: {counts_path}\n")
        f.write(f"Samples: {n_samples}, Features: {n_features}, Events: {n_events}\n")
        f.write(f"EPV: {n_events / n_features:.2f}\n\n")
        f.write(f"Min FDR+: {stabl.min_fdr_:.4f}\n")
        f.write(f"FDR Threshold: {stabl.fdr_min_threshold_:.4f}\n\n")
        f.write(f"Real features - Mean: {np.mean(max_scores_real):.4f}, Std: {np.std(max_scores_real):.4f}\n")
        f.write(f"Artificial features - Mean: {np.mean(max_scores_artificial):.4f}, Std: {np.std(max_scores_artificial):.4f}\n")
        f.write(f"Effect size: {effect_size:.4f}\n")

    print(f"Report saved: {report_path}")

    return stabl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose FDP+ issues in STABL Cox")
    parser.add_argument("--counts", required=True, help="Path to counts file")
    parser.add_argument("--clinical", required=True, help="Path to clinical file")
    parser.add_argument("--num_genes", type=int, default=100, help="Number of genes to use")
    parser.add_argument("--n_bootstraps", type=int, default=100, help="Number of bootstraps (use fewer for quick diagnosis)")
    parser.add_argument("--outdir", default="./fdp_diagnostics", help="Output directory")

    args = parser.parse_args()

    diagnose_fdp_issue(
        counts_path=args.counts,
        clinical_path=args.clinical,
        num_genes=args.num_genes,
        n_bootstraps=args.n_bootstraps,
        output_dir=args.outdir
    )
