#%% [Markdown]
# ## Preliminaries


#%% [code]
# Includes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from groupBMC.groupBMC import GroupBMC

np.set_printoptions(suppress=True,precision=3)


#%% [code]
# Load data

file_path = "summary_subject_x_choice_counts.csv"
    
df_all = pd.read_csv(file_path, dtype='int64')
#df_all = df_all.head(3)


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
assert all(df_counts.sum(axis=1) == 27), "Not all rows sum to 27"

# Get the number of options as <n, k>
df_num_options = pd.DataFrame(np.tile(num_options, (len(df_counts), 1)), columns=df_counts.columns)

counts = df_counts.to_numpy()

#%% [code]
# Multinomial mixture EM algorithm

def multinomial_mixture_EM(counts, n_clusters=2, max_iter=200, tol=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    S, K = counts.shape
    N = counts.sum(axis=1, keepdims=True)
    # initialize cluster probabilities randomly
    theta = rng.dirichlet(np.ones(K), size=n_clusters)
    pi = np.ones(n_clusters) / n_clusters

    for it in range(max_iter):
        # --- E-step ---
        loglik = np.zeros((S, n_clusters))
        for j in range(n_clusters):
            loglik[:, j] = np.log(pi[j]) + (counts * np.log(theta[j] + 1e-12)).sum(axis=1)
        loglik -= loglik.max(axis=1, keepdims=True)  # stability
        r = np.exp(loglik)
        r /= r.sum(axis=1, keepdims=True)

        # --- M-step ---
        pi_new = r.mean(axis=0)
        theta_new = np.zeros_like(theta)
        for j in range(n_clusters):
            w = r[:, j:j+1]
            theta_new[j] = (w.T @ counts).ravel()
        theta_new /= theta_new.sum(axis=1, keepdims=True)

        # convergence check
        diff = np.abs(theta_new - theta).max()
        theta, pi = theta_new, pi_new
        if diff < tol:
            break

    return pi, theta, r

#%% [code]
# Run the EM algorithm

pi, theta, r = multinomial_mixture_EM(df_counts.to_numpy(), n_clusters=3)

print("Cluster probabilities:", pi)
print("Cluster means:", theta)
print("Responsibilities:", r)

#%% [code]
# Plot the results