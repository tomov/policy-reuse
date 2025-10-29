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

np.set_printoptions(suppress=True, precision=3)


#%% [code]
# Load data

from load_data import load_data

data = load_data()
df_counts = data['df_counts']

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