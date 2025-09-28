#!/usr/bin/env python3
"""
Script to load and analyze the summary_subject_x_choice_counts.csv data.
This data contains choice counts for different policy reuse strategies across subjects.
"""

#%% [markdown]
# ## Preliminaries


#%% [code]
# Includes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#%% [code]
# Load data

file_path = "policy-composition-momchil/Output/V0.3_pilot/files/summaries/summary_subject_x_choice_counts.csv"
    
df_all = pd.read_csv(file_path, dtype='int64')
df_all = df_all.head(3)



# %% [code]
# Transform into mutually exclusive choices

# Create the new column first, then filter
df_all['policy reuse min rew. test'] = df_all['policy reuse cued'] - df_all['policy reuse max rew. test']

choice_columns = ['gpi zero', 
                  'policy reuse max rew. test', 
                  'policy reuse min rew. test',
                  'policy reuse uncued', 
                  'mb/gpi', 
                  'null trajectories']
num_options = [1, 1, 1, 1, 1, 4]

df_counts = df_all[choice_columns]
df_num_options = pd.DataFrame(np.tile(num_options, (len(df_counts), 1)), columns=df_counts.columns)

# %% [code]
# Sanity check that all columns sum to 27 == the number of test trials per subject

df_totals = df_counts.sum(axis=1)
assert all(df_totals == 27), f"Not all rows sum to 27. Found sums: {df_totals.unique()}"


# %%

def get_proportions(column_names: list[str]):
    counts = df_counts[column_names].sum(axis=1)
    proportions = counts / df_counts.sum(axis=1) / df_num_options[column_names].sum(axis=1)
    return proportions

class Hypothesis:
    
    def __init__(self, name: str, column_groups: dict[str, list[str]]):
        # Assume uniform choices in each column group
        self.name = name
        self.column_groups = column_groups
        assigned_columns = {col for columns in column_groups.values() for col in columns}
        self.column_groups['e'] = [col for col in choice_columns if col not in assigned_columns]
        
    def fit(self):
        self.proportions = {group_name: get_proportions(columns) for group_name, columns in self.column_groups.items()}
        
        # Map proportions to corresponding columns
        self.column_proportions = np.zeros(len(choice_columns))
        for group_name, columns in self.column_groups.items():
            group_prop = self.proportions[group_name].mean()
            for col in columns:
                self.column_proportions[choice_columns.index(col)] = group_prop

        
# H0: [e,e,e,e,e,4e]
H0 = Hypothesis("Uniform", {})

# H1: [e,p,q,r,e,4e]
H1 = Hypothesis("Policy reuse", {'p': ['policy reuse max rew. test'], 
                                 'q': ['policy reuse min rew. test'], 
                                 'r': ['policy reuse uncued']})

# H2: [e,p,p,p,e,4e]
H2 = Hypothesis("Policy reuse uniform", {'p': ['policy reuse max rew. test','policy reuse min rew. test','policy reuse uncued']})

# H3: [e,p,p,e,e,4e]
#H2 = Hypothesis("Policy reuse cued", ['policy reuse max rew. test', 'policy reuse min rew. test'])
#H3 = Hypothesis("Policy reuse best", ['policy reuse max rew. test'])
#H3 = Hypothesis("Policy reuse best", ['policy reuse max rew. test'])

# %%

mle_parameters = {group_name: get_proportions(columns) for group_name, columns in H2.column_groups.items()}

df_proportions = pd.DataFrame(np.zeros(df_counts.shape, dtype=float), columns=df_counts.columns)
for group_name, column_names in H2.column_groups.items():
    tiled_proportions = np.tile(mle_parameters[group_name].values[:, np.newaxis], (1, len(column_names)))
    df_proportions[column_names] = tiled_proportions    
    df_proportions *= df_num_options
    
assert np.allclose(df_proportions.sum(axis=1), 1.0), f"Rows don't sum to 1: {df_proportions.sum(axis=1)}"

log_likelihood = np.sum(df_counts * np.log(df_proportions))
    
# %%
