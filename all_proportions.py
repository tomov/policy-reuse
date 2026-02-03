"""
Visualize proportions of test trial choices across all experiment versions.
Generates a grid of subplots, one per experiment, showing mean proportions
with SEM error bars and per-category chance levels.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from load_data import load_full_data_for_experiment

# All experiment versions with results
experiment_versions = [
    "V0.3_pilot", "V0.4_pilot", "V0.5_pilot", "V0.6_pilot",
    "V0.7_pilot", "V0.8_pilot", "V0.9_pilot", "V1.0_pilot",
    "V1.1_pilot", "V1.2_pilot", "V1.3_pilot", "V1.4_pilot",
    "V1.5_pilot", "V1.6_pilot",
]

# Load experiment summary for metadata (n_included)
summary = pd.read_csv("results/experiment_versions_summary.csv")
summary = summary.set_index("version")

# Unique choice columns (mutually exclusive bins)
UNIQUE_COLS = [
    'gpi zero',
    'policy reuse max rew. test',
    'policy reuse min rew. test',
    'mb/gpi',
    'null trajectories',
]

# Short display labels
LABELS = {
    'gpi zero': 'GPI-0',
    'policy reuse max rew. test': 'PR max',
    'policy reuse min rew. test': 'PR min',
    'mb/gpi': 'MB/GPI',
    'null trajectories': 'Null',
}

# Consistent colors per category
COLORS = {
    'gpi zero': '#ff7f0e',
    'policy reuse max rew. test': '#2ca02c',
    'policy reuse min rew. test': '#9467bd',
    'mb/gpi': '#1f77b4',
    'null trajectories': '#d62728',
}

# Grid layout
n_exp = len(experiment_versions)
ncols = 4
nrows = int(np.ceil(n_exp / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows),
                         sharey=True)
axes = axes.flatten()

for idx, version in enumerate(experiment_versions):
    ax = axes[idx]

    try:
        data = load_full_data_for_experiment(version)
    except Exception as e:
        ax.set_title(version, fontsize=10, fontweight='bold')
        ax.text(0.5, 0.5, f"Load error:\n{e}", transform=ax.transAxes,
                ha='center', va='center', fontsize=7, color='red')
        continue

    df_counts = data['df_counts']
    df_num_options = data['df_num_options']

    # Only plot columns that exist in this experiment
    cols = [c for c in UNIQUE_COLS if c in df_counts.columns]
    if not cols:
        ax.set_title(version, fontsize=10, fontweight='bold')
        ax.text(0.5, 0.5, "No matching columns", transform=ax.transAxes,
                ha='center', va='center', fontsize=8)
        continue

    # Proportions: divide each count by total (random choice = total test trials)
    proportions = df_counts[cols].div(df_counts['random choice'], axis=0)
    means = proportions.mean(axis=0)
    sems = proportions.std(axis=0) / np.sqrt(len(proportions))

    # Chance levels
    total_options = df_num_options.iloc[0]['random choice']
    chance = df_num_options.iloc[0][cols] / total_options

    x_pos = np.arange(len(cols))
    colors = [COLORS[c] for c in cols]
    labels = [LABELS[c] for c in cols]

    ax.bar(x_pos, means.values, yerr=sems.values,
           capsize=3, alpha=0.85, color=colors,
           edgecolor='black', linewidth=0.6, width=0.7)

    # Chance markers
    for i, (x, cv) in enumerate(zip(x_pos, chance.values)):
        ax.plot([x - 0.35, x + 0.35], [cv, cv],
                color='red', linestyle='--', linewidth=1.2, alpha=0.6)

    # Title with sample size
    ver_key = version.replace("_pilot", "")
    n_inc = ""
    if ver_key in summary.index:
        n_inc = f"  (n={int(summary.loc[ver_key, 'n_included'])})"

    ax.set_title(f"{version}{n_inc}", fontsize=10, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 0.7)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.grid(axis='y', alpha=0.25, linewidth=0.5)
    ax.tick_params(axis='y', labelsize=8)

# Label shared y-axis on leftmost panels
for row in range(nrows):
    axes[row * ncols].set_ylabel('Proportion', fontsize=9)

# Hide unused axes and use one for the legend
for idx in range(n_exp, len(axes)):
    axes[idx].set_visible(False)

if n_exp < len(axes):
    legend_ax = axes[n_exp]
    legend_ax.set_visible(True)
    legend_ax.axis('off')
    handles = [Patch(facecolor=COLORS[c], edgecolor='black', linewidth=0.6,
                     label=LABELS[c]) for c in UNIQUE_COLS]
    handles.append(Line2D([0], [0], color='red', linestyle='--',
                          linewidth=1.2, alpha=0.6, label='Chance'))
    legend_ax.legend(handles=handles, loc='center', fontsize=10, frameon=True,
                     title='Choice Category', title_fontsize=11)

fig.suptitle('Proportion of Test Trial Choices Across All Experiments',
             fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig('results/all_proportions.png', dpi=180, bbox_inches='tight')
plt.show()
print("Saved to results/all_proportions.png")
