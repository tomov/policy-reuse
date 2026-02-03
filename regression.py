#%% [markdown]
# ## Log-odds regression: experiment features → choice behavior

#%% [code]
# Includes

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from load_data import load_data_for_experiment, UNIQUE_CHOICE_COLUMNS, UNIQUE_CHOICE_COLUMNS_NO_UNCUED

np.set_printoptions(suppress=True, precision=3)

#%% [code]
# Configuration

# Experiment versions to include (all except V0.1 and V0.2)
THREE_TASK_VERSIONS = ["V0.3_pilot", "V0.4_pilot", "V0.5_pilot", "V0.6_pilot",
                       "V0.7_pilot", "V0.8_pilot", "V0.9_pilot", "V1.0_pilot",
                       "V1.1_pilot", "V1.3_pilot"]
TWO_TASK_VERSIONS = ["V1.2_pilot", "V1.4_pilot", "V1.5_pilot", "V1.6_pilot"]
ALL_VERSIONS = THREE_TASK_VERSIONS + TWO_TASK_VERSIONS

# Pairwise log-odds contrasts (numerator, denominator)
LOG_ODDS_PAIRS = [
    ("gpi zero", "policy reuse max rew. test"),
    ("gpi zero", "mb/gpi"),
    ("policy reuse max rew. test", "policy reuse min rew. test"),
]

# Predictor columns from experiment_versions_summary.csv
PREDICTOR_COLUMNS = [
    "n_tasks",
    #"n_blocks",
    "train_trials_per_block",
    "grid_states",
    "total_trajectories",
    "design_type",
    #"max_pellets",
]

# Pseudocount added to proportions before computing log-odds
PSEUDOCOUNT = 0.1

#%% [code]
# Load subject-level counts and experiment-level features

# Load experiment metadata
meta = pd.read_csv("results/experiment_versions_summary.csv")
meta = meta.rename(columns={"version": "experiment_version"})
# Add _pilot suffix to match version strings
meta["experiment_version"] = meta["experiment_version"] + "_pilot"
meta = meta[meta["experiment_version"].isin(ALL_VERSIONS)]

# Load subject-level data for each experiment
rows = []
for version in ALL_VERSIONS:
    choice_cols = UNIQUE_CHOICE_COLUMNS if version in THREE_TASK_VERSIONS else UNIQUE_CHOICE_COLUMNS_NO_UNCUED
    data = load_data_for_experiment(version, choice_columns=choice_cols)
    counts = data["df_counts"]
    random_choice = int(data["df_all"]["random choice"].iloc[0])

    for i in range(len(counts)):
        row = {"experiment_version": version, "subject": i, "random_choice": random_choice}
        for col in counts.columns:
            row[col] = counts.iloc[i][col]
        rows.append(row)

df_subjects = pd.DataFrame(rows)

# Merge with experiment metadata
df = df_subjects.merge(meta, on="experiment_version", how="left")

print(f"Total subjects: {len(df)}")
print(f"Experiments: {df['experiment_version'].nunique()}")
print(df.groupby("experiment_version").size())

#%% [code]
# Compute log-odds for each contrast

for num_col, den_col in LOG_ODDS_PAIRS:
    # Normalize counts to proportions, add pseudocount
    num = df[num_col] / df["random_choice"] + PSEUDOCOUNT
    den = df[den_col] / df["random_choice"] + PSEUDOCOUNT
    col_name = f"logodds_{num_col}_vs_{den_col}".replace(" ", "_").replace(".", "").replace("/", "_")
    df[col_name] = np.log(num / den)

logodds_columns = [c for c in df.columns if c.startswith("logodds_")]
print("Log-odds columns:")
for c in logodds_columns:
    print(f"  {c}: mean={df[c].mean():.3f}, std={df[c].std():.3f}")

#%% [code]
# Run OLS regressions with clustered standard errors by experiment

results = {}
for logodds_col in logodds_columns:
    # Build formula
    predictors = []
    for col in PREDICTOR_COLUMNS:
        if df[col].dtype == object or col == "design_type":
            predictors.append(f"C({col})")
        else:
            predictors.append(col)
    formula = f"{logodds_col} ~ " + " + ".join(predictors)

    model = smf.ols(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["experiment_version"]},
    )
    results[logodds_col] = model

    print(f"\n{'='*80}")
    print(f"Dependent variable: {logodds_col}")
    print(f"{'='*80}")
    print(model.summary())

#%% [code]
# Summary table of significant predictors

print(f"\n{'='*80}")
print("Summary of significant predictors (p < 0.05)")
print(f"{'='*80}")

for logodds_col, model in results.items():
    print(f"\n{logodds_col}:")
    sig = model.pvalues[model.pvalues < 0.05]
    if len(sig) == 0:
        print("  (none)")
    else:
        for name, pval in sig.items():
            coef = model.params[name]
            print(f"  {name}: coef={coef:.4f}, p={pval:.4f}")

#%% [code]