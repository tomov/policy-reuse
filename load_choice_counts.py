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
    
df = pd.read_csv(file_path, dtype='int64')


# %% [code]
# Transform into mutually exclusive choices

# Create the new column first, then filter
df['policy reuse min rew. test'] = df['policy reuse cued'] - df['policy reuse max rew. test']
df_exclusive = df[['gpi zero', 
                   'policy reuse max rew. test', 
                   'policy reuse min rew. test',
                   'policy reuse uncued', 
                   'mb/gpi', 
                   'null trajectories']]

# %% [code]
# Sanity check that all columns sum to 27 == the number of test trials per subject

df_totals = df_exclusive.sum(axis=1)
assert all(df_totals == 27), f"Not all rows sum to 27. Found sums: {df_totals.unique()}"


# %%
