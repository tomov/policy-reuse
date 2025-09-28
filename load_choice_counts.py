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

# Key columns (bins) == option categories, such that 1) they cover all options and 2) they are mutually exclusive
choice_columns = ['gpi zero', 
                  'policy reuse max rew. test', 
                  'policy reuse min rew. test',
                  'policy reuse uncued', 
                  'mb/gpi', 
                  'null trajectories']
# Number of options in each column/bin
num_options = [1, 1, 1, 1, 1, 4]

# Get the empirical choice counts as <n, k>, where n is the number of subjects and k is the number of columns
df_counts = df_all[choice_columns]

# Get the number of options as <n, k>
df_num_options = pd.DataFrame(np.tile(num_options, (len(df_counts), 1)), columns=df_counts.columns)

# %% [code]
# Sanity check that all columns sum to 27 == the number of test trials per subject

df_totals = df_counts.sum(axis=1)
assert all(df_totals == 27), f"Not all rows sum to 27. Found sums: {df_totals.unique()}"


# %% [code]
# Simple multinomial parameter fitting in closed form

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
    """ Multinomial parameter """
    
    def __init__(self, name: str, column_names: list[str], limits: tuple[float, float]):
        """
        :param name: the name of the parameter
        :param column_names: the columns that the parameter encompasses
        :param limits: the limits of the parameter as a tuple of (lower, upper)
        """
        self.name = name
        self.column_names = column_names
        self.limits = limits
        self.df_mle_value = None
        
    def fit(self):
        """ Fit the parameter """
        self.df_mle_value = get_probability_of_uniformly_choosing_among(self.column_names)
        self.df_mle_value = self.df_mle_value.clip(lower=self.limits[0], upper=self.limits[1])

class Hypothesis:
    """ Simple hypothesis about the multinomial distribution parameters """
    
    def __init__(self, name: str, parameters: list[Parameter]):
        """
        :param name: the name of the hypothesis
        :param parameters: the parameters that the hypothesis encompasses
        """
        self.name = name
        self.parameters = parameters
        self.df_column_probabilities = None
        self.log_likelihood = None
        
        # Add noise parameter for all columns that are not assigned to a parameter
        assigned_columns = set().union(*[param.column_names for param in parameters])
        self.parameters.append(Parameter('e', [col for col in choice_columns if col not in assigned_columns], (0, 1/9)))
        
    def fit(self):
        """ Fit the hypothesis """
        
        # Get the column probabilities as <n, k> where k is the number of columns
        self.df_column_probabilities = pd.DataFrame(np.zeros(df_counts.shape, dtype=float), columns=df_counts.columns)
        for parameter in self.parameters:
            parameter.fit()
            tiled_proportions = np.tile(parameter.df_mle_value.values[:, np.newaxis], (1, len(parameter.column_names)))
            self.df_column_probabilities[parameter.column_names] = tiled_proportions    
            self.df_column_probabilities *= df_num_options
        assert np.allclose(df_proportions.sum(axis=1), 1.0), f"Rows don't sum to 1: {df_proportions.sum(axis=1)}"

        # Get the log likelihood as <n, 1>
        self.log_likelihood = np.sum(df_counts * np.log(self.df_column_probabilities), axis=1)

# %% [code]
# Define hypotheses
        
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

# H3: [e,p,q,e,e,4e]
H1 = Hypothesis("Policy reuse cued", 
                [Parameter('p', ['policy reuse max rew. test'], (1/9, 1)), 
                 Parameter('q', ['policy reuse min rew. test'], (1/9, 1))])

# H3: [e,p,p,e,e,4e]
H3 = Hypothesis("Policy reuse cued uniform", 
                [Parameter('p', ['policy reuse max rew. test', 'policy reuse min rew. test'], (1/9, 1))])

# H4: [e,p,e,e,e,4e]
H4 = Hypothesis("Policy reuse best", 
                [Parameter('p', ['policy reuse max rew. test'], (1/9, 1))])

# H5: [e,e,e,e,p,4e]
H5 = Hypothesis("MB/GPI", 
                [Parameter('p', ['mb/gpi'], (1/9, 1))])

# H6: [p,e,e,e,e,4e]
H6 = Hypothesis("GPI zero", 
                [Parameter('p', ['gpi zero'], (1/9, 1))])

hypotheses = [H0, H1, H2, H3, H4, H5, H6]

# %% [code]
# Fit the hypotheses

[H.fit() for H in hypotheses]

log_likelihoods = np.column_stack([H.log_likelihood for H in hypotheses])

    
# %%
