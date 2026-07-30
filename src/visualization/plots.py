"""
HCC Delta Vasculomics - Visualization Module
=============================================
Publication-quality figures for the delta vasculomics study.

Figure types:
- Patient flow chart
- Clinical characteristics table
- QVMF fingerprint heatmap (matching Fig. 2 of original)
- Paired t-test bar charts
- ROC curves with CV
- SHAP visualizations (bar, summary, waterfall, dependence)
- Feature correlation networks
- Calibration curves
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Style settings for publication
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color scheme matching original paper
COLORS = {
    'good': '#E74C3C',      # Pink-red for good responders
    'poor': '#3498DB',      # Light blue for poor responders
    'significant': '#2C3E50',
    'non_significant': '#BDC3C7',
}

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "figures"


def plot_morphological_fingerprint(
    feature_df: pd.DataFrame,
    response_col: str = 'response',
    feature_groups: Optional[Dict[str, List[str]]] = None,
    max_features: int = 20,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
):
    """
    Plot morphological fingerprint heatmap (z-score).
    Matching Fig. 2 of Hu et al. (2026).
    """
    poor = feature_df[feature_df[response_col] == 0]
    good = feature_df[feature_df[response_col] == 1]

    # Select features
    feature_cols = [c for c in feature_df.columns
                    if c != response_col and c != 'patient_id'
                    and feature_df[c].dtype in ['float64', 'float32', 'int64']]

    if max_features and len(feature_cols) > max_features:
        # Select top features by univariate significance
        pvals = {}
        for col in feature_cols:
            try:
                _, p = stats.mannwhitneyu(
                    poor[col].dropna(), good[col].dropna(),
                    alternative='two-sided'
                )
                pvals[col] = p
            except Exception:
                pvals[col] = 1.0
        feature_cols = sorted(pvals, key=pvals.get)[:max_features]

    # Compute z-scores (using pooled mean/std)
    z_scores_poor = {}
    z_scores_good = {}
    p_values = {}

    for col in feature_cols:
        vals_all = feature_df[col].dropna()
        if len(vals_all) < 3:
            continue
        mean_all = vals_all.mean()
        std_all = vals_all.std()
        if std_all == 0:
            continue

        z_scores_poor[col] = (poor[col].mean() - mean_all) / std_all
        z_scores_good[col] = (good[col].mean() - mean_all) / std_all

        try:
            _, p = stats.mannwhitneyu(
                poor[col].dropna(), good[col].dropna(),
                alternative='two-sided'
            )
            p_values[col] = p
        except Exception:
            p_values[col] = 1.0

    # Sort features by effect size difference
    features_sorted = sorted(
        z_scores_poor.keys(),
        key=lambda c: abs(z_scores_good[c] - z_scores_poor[c]),
        reverse=True
    )

    if not features_sorted:
        warnings.warn("No valid features for fingerprint plot.")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    y_pos = range(len(features_sorted))

    # Bar chart showing z-scores
    poor_vals = [z_scores_poor[f] for f in features_sorted]
    good_vals = [z_scores_good[f] for f in features_sorted]

    bar_height = 0.35
    ax.barh([y + bar_height/2 for y in y_pos], poor_vals, bar_height,
            color=COLORS['poor'], alpha=0.8, label='Poor Response')
    ax.barh([y - bar_height/2 for y in y_pos], good_vals, bar_height,
            color=COLORS['good'], alpha=0.8, label='Good Response')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features_sorted, fontsize=7)
    ax.set_xlabel('Morphological Fingerprint (z-score)')
    ax.axvline(x=0, color='black', linewidth=0.5, linestyle='--')
    ax.legend(loc='lower right')
    ax.set_title('Vascular Morphological Fingerprint: Good vs Poor Responders')

    # Add P-value annotations
    for i, feat in enumerate(features_sorted):
        p = p_values[feat]
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        if sig:
            max_val = max(abs(poor_vals[i]), abs(good_vals[i]))
            ax.text(max_val + 0.1, i, f'{sig}\nP={p:.3f}',
                   fontsize=5, va='center', color=COLORS['significant'])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig


def plot_paired_boxplots(
    feature_df: pd.DataFrame,
    feature_name: str,
    response_col: str = 'response',
    paired_col: str = 'timepoint',
    timepoint_labels: Tuple[str, str] = ('Baseline', 'Follow-up'),
    ax: Optional[plt.Axes] = None,
):
    """
    Paired boxplot showing change in a feature between two timepoints,
    stratified by response group.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))

    poor = feature_df[feature_df[response_col] == 0]
    good = feature_df[feature_df[response_col] == 1]

    data_poor = [
        poor[poor[paired_col] == timepoint_labels[0]][feature_name].values,
        poor[poor[paired_col] == timepoint_labels[1]][feature_name].values,
    ]
    data_good = [
        good[good[paired_col] == timepoint_labels[0]][feature_name].values,
        good[good[paired_col] == timepoint_labels[1]][feature_name].values,
    ]

    positions = [1, 2, 4, 5]
    bp1 = ax.boxplot(data_poor, positions=[1, 2], widths=0.6,
                     patch_artist=True, showfliers=False)
    bp2 = ax.boxplot(data_good, positions=[4, 5], widths=0.6,
                     patch_artist=True, showfliers=False)

    for patch in bp1['boxes']:
        patch.set_facecolor(COLORS['poor'])
    for patch in bp2['boxes']:
        patch.set_facecolor(COLORS['good'])

    ax.set_xticks([1.5, 4.5])
    ax.set_xticklabels(['Poor Response', 'Good Response'])
    ax.set_ylabel(feature_name)
    ax.set_title(f'{feature_name} Change')

    # Add P-value
    if len(data_poor[0]) > 0 and len(data_poor[1]) > 0:
        try:
            _, p = stats.wilcoxon(data_poor[0], data_poor[1])
            ax.text(1.5, ax.get_ylim()[1] * 0.95, f'P={p:.3f}', ha='center', fontsize=7)
        except Exception:
            pass

    return ax


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    ax: Optional[plt.Axes] = None,
    label: str = 'Model',
    color: str = '#2C3E50',
):
    """
    Plot calibration curve with reliability diagram.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Bin predictions
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_fractions = []

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append(y_prob[mask].mean())
            bin_fractions.append(y_true[mask].mean())
        else:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_fractions.append(np.nan)

    ax.plot(bin_centers, bin_fractions, 'o-', color=color, label=label,
           markersize=5, linewidth=1.5)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5, label='Perfect calibration')

    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Observed Fraction')
    ax.set_title('Calibration Curve')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return ax


def plot_univariate_forest(
    feature_df: pd.DataFrame,
    response_col: str = 'response',
    feature_cols: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
):
    """
    Forest plot of univariate odds ratios for each feature.
    """
    from sklearn.linear_model import LogisticRegression

    poor = feature_df[feature_df[response_col] == 0]
    good = feature_df[feature_df[response_col] == 1]

    if feature_cols is None:
        feature_cols = [c for c in feature_df.columns
                        if c not in [response_col, 'patient_id']
                        and feature_df[c].dtype in ['float64', 'float32']]

    ors = {}
    cis = {}
    pvals = {}

    for col in feature_cols:
        # Standardize
        mean = feature_df[col].mean()
        std = feature_df[col].std()
        if std == 0:
            continue

        X = (feature_df[col].values.reshape(-1, 1) - mean) / std
        y = feature_df[response_col].values

        try:
            lr = LogisticRegression(penalty=None, max_iter=1000)
            lr.fit(X, y)
            or_val = np.exp(lr.coef_[0][0])
            ors[col] = or_val

            # CI via bootstrap
            boot_ors = []
            n = len(X)
            for _ in range(200):
                idx = np.random.choice(n, n, replace=True)
                lr_boot = LogisticRegression(penalty=None, max_iter=1000)
                lr_boot.fit(X[idx], y[idx])
                boot_ors.append(np.exp(lr_boot.coef_[0][0]))
            ci_low = np.percentile(boot_ors, 2.5)
            ci_high = np.percentile(boot_ors, 97.5)
            cis[col] = (ci_low, ci_high)
            pvals[col] = stats.mannwhitneyu(
                poor[col].dropna(), good[col].dropna()
            ).pvalue
        except Exception:
            continue

    # Sort by OR
    features_sorted = sorted(ors.keys(), key=lambda c: abs(np.log(ors[c])))

    fig, ax = plt.subplots(figsize=figsize)

    y_positions = range(len(features_sorted))
    colors_list = [COLORS['significant'] if pvals[f] < 0.05
                   else COLORS['non_significant'] for f in features_sorted]

    for i, feat in enumerate(features_sorted):
        or_val = ors[feat]
        ci_low, ci_high = cis[feat]

        ax.errorbar(or_val, i, xerr=[[or_val - ci_low], [ci_high - or_val]],
                   fmt='o', color=colors_list[i], capsize=3, markersize=6)
        ax.axvline(x=1, color='black', linewidth=0.5, linestyle='--')

        sig = '***' if pvals[feat] < 0.001 else ('**' if pvals[feat] < 0.01 else ('*' if pvals[feat] < 0.05 else ''))
        label = f"{feat} {sig}"

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([f"{f} ({ors[f]:.2f}, {cis[f][0]:.2f}-{cis[f][1]:.2f})"
                        for f in features_sorted], fontsize=6)
    ax.set_xlabel('Odds Ratio (per SD)')
    ax.set_title('Univariate Analysis: Odds Ratios for Response')
    ax.set_xscale('log')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig


def plot_clinical_table(
    clin_df: pd.DataFrame,
    response_col: str = 'response',
    save_path: Optional[str] = None,
):
    """
    Generate a formatted clinical characteristics table as an image.
    Matching Table 1 of the original paper.
    """
    poor = clin_df[clin_df[response_col] == 0]
    good = clin_df[clin_df[response_col] == 1]

    table_data = []
    exclude_cols = [response_col, 'patient_id']

    for col in clin_df.columns:
        if col in exclude_cols:
            continue

        if clin_df[col].dtype in ['float64', 'float32', 'int64']:
            mean_poor = poor[col].mean()
            std_poor = poor[col].std()
            mean_good = good[col].mean()
            std_good = good[col].std()

            # P-value
            try:
                _, p = stats.mannwhitneyu(poor[col].dropna(), good[col].dropna())
            except Exception:
                p = np.nan

            table_data.append([
                col,
                f"{mean_poor:.1f}±{std_poor:.1f}",
                f"{mean_good:.1f}±{std_good:.1f}",
                f"{p:.3f}" if not np.isnan(p) else "N/A"
            ])
        else:
            # Categorical
            poor_counts = poor[col].value_counts()
            good_counts = good[col].value_counts()

            for cat in clin_df[col].unique():
                p_poor = poor_counts.get(cat, 0) / len(poor) * 100
                p_good = good_counts.get(cat, 0) / len(good) * 100

                try:
                    ct = pd.crosstab(
                        clin_df[col] == cat,
                        clin_df[response_col]
                    )
                    from scipy.stats import chi2_contingency
                    _, p, _, _ = chi2_contingency(ct)
                except Exception:
                    p = np.nan

                table_data.append([
                    f"{col} - {cat}",
                    f"{poor_counts.get(cat, 0)} ({p_poor:.1f}%)",
                    f"{good_counts.get(cat, 0)} ({p_good:.1f}%)",
                    f"{p:.3f}" if not np.isnan(p) else "N/A"
                ])

    df_table = pd.DataFrame(
        table_data,
        columns=['Characteristic', 'Poor (n={})'.format(len(poor)),
                 'Good (n={})'.format(len(good)), 'P-value']
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_table.to_csv(save_path, index=False)
        print(f"Saved: {save_path}")

    return df_table


if __name__ == "__main__":
    print("Visualization module initialized.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
