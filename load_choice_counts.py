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

def get_probability_of_uniformly_choosing_among(column_names: list[str]) -> pd.DataFrame:
    """Get the MLE probability (i.e. empirical frequency) of choosing one of options in the given columns, assuming it is the same for all of them.
    
    Specifically, we collapse all options into two bins: the options in the given columns and the rest.
    Then we get P(choosing options in the given columns) = # choices in the given columns / # total choices / # options in the given columns
    
    Note that multiple options can correspond to a single column (e.g. the "null trajectories" column corresponds to 4 options)
    
    For example, if we have 4 columns ('A', 'B', 'C', 'D') with option counts [1, 1, 1, 4] and choice counts [10, 20, 15, 5], 
    then for single columns, we have:
    - The answer for column_names = ['A'] is 10 / (10 + 20 + 15 + 5) / 1 = 10 / 50 = 0.2
    - The answer for column_names = ['B'] is 20 / (10 + 20 + 15 + 5) / 1 = 20 / 50 = 0.4  
    - The answer for column_names = ['C'] is 15 / (10 + 20 + 15 + 5) / 1 = 15 / 50 = 0.3  
    - The answer for column_names = ['D'] is 5 / (10 + 20 + 15 + 5) / 4 = 5 / 50 / 4 = 0.025 
    * Note that A + B + C + 4 * D = 1.0 

    For multiple columns, we have:
    - The answer for column_names = ['A', 'B'] is (10 + 20) / (10 + 20 + 15 + 5) / (1 + 1) = 30 / 50 / 2 = 0.3
    - The answer for column_names = ['B', 'C', 'D'] is (20 + 15 + 5) / (10 + 20 + 15 + 5) / (1 + 1 + 4) = 40 / 50 / 6 = 0.133333
    - The answer for column_names = ['A', 'B', 'C', 'D'] is (10 + 20 + 15 + 5) / (10 + 20 + 15 + 5) / (1 + 1 + 1 + 4) = 50 / 50 / 7 = 1/7 = 0.142857
    
    :param column_names: the columns of interest
    :return: <n, 1> the MLE probability of the options in the column names, assuming it is the same for all of them
    """
    counts = df_counts[column_names].sum(axis=1)
    proportions = counts / df_counts.sum(axis=1) / df_num_options[column_names].sum(axis=1)
    return proportions

class Parameter:
    def __init__(self, name: str, column_names: list[str], limits: tuple[float, float]):
        self.name = name
        self.column_names = column_names
        self.limits = limits
        self.df_mle_value = None
        
    def fit(self):
        self.df_mle_value = get_probability_of_uniformly_choosing_among(self.column_names)

class Hypothesis:
    
    def __init__(self, name: str, parameters: list[Parameter]):
        self.name = name
        self.parameters = parameters
        # Add noise parameter for all columns that are not assigned to a parameter
        assigned_columns = set().union(*[param.column_names for param in parameters])
        self.parameters.append(Parameter('e', [col for col in choice_columns if col not in assigned_columns], (0, 1/9)))
        
    def fit(self):
        self.proportions = {group_name: get_proportions(columns) for group_name, columns in self.column_groups.items()}
        
        # Map proportions to corresponding columns
        self.column_proportions = np.zeros(len(choice_columns))
        for group_name, columns in self.column_groups.items():
            group_prop = self.proportions[group_name].mean()
            for col in columns:
                self.column_proportions[choice_columns.index(col)] = group_prop

        
# H0: [e,e,e,e,e,4e]
H0 = Hypothesis("Uniform", [])

# H1: [e,p,q,r,e,4e]
H1 = Hypothesis("Policy reuse", 
                [Parameter('p', ['policy reuse max rew. test'], (1/9, 1)), 
                 Parameter('q', ['policy reuse min rew. test'], (1/9, 1)), 
                 Parameter('r', ['policy reuse uncued'], (1/9, 1))])

# H2: [e,p,p,p,e,4e]
H2 = Hypothesis("Policy reuse uniform", 
                [Parameter('p', ['policy reuse max rew. test','policy reuse min rew. test','policy reuse uncued'], (1/9, 1))])

# H3: [e,p,p,e,e,4e]
#H2 = Hypothesis("Policy reuse cued", ['policy reuse max rew. test', 'policy reuse min rew. test'])
#H3 = Hypothesis("Policy reuse best", ['policy reuse max rew. test'])
#H3 = Hypothesis("Policy reuse best", ['policy reuse max rew. test'])

# %%

df_proportions = pd.DataFrame(np.zeros(df_counts.shape, dtype=float), columns=df_counts.columns)
for parameter in H2.parameters:
    parameter.fit()
    tiled_proportions = np.tile(parameter.df_mle_value.values[:, np.newaxis], (1, len(parameter.column_names)))
    df_proportions[parameter.column_names] = tiled_proportions    
    df_proportions *= df_num_options
    
assert np.allclose(df_proportions.sum(axis=1), 1.0), f"Rows don't sum to 1: {df_proportions.sum(axis=1)}"

log_likelihood = np.sum(df_counts * np.log(df_proportions))
    
# %%
