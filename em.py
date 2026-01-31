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
from scipy.special import logsumexp

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

    # compute total log-likelihood
    loglik_final = np.zeros((S, n_clusters))
    for j in range(n_clusters):
        loglik_final[:, j] = np.log(pi[j]) + (counts * np.log(theta[j] + 1e-12)).sum(axis=1)
    max_ll = loglik_final.max(axis=1, keepdims=True)
    ll = np.sum(max_ll.ravel() + np.log(np.exp(loglik_final - max_ll).sum(axis=1)))

    return pi, theta, r, ll

#%% [code]
# Run the EM algorithm

pi, theta, r, ll = multinomial_mixture_EM(df_counts.to_numpy(), n_clusters=4)

print("Cluster probabilities:", pi)
print("Cluster means:", theta)
print("Responsibilities:", r)

#%% [code]
# Fit models for a range of cluster counts

counts = df_counts.to_numpy()
S, K = counts.shape
cluster_range = range(1, 10)

results = {}
for C in cluster_range:
    print(f"Fitting model with {C} clusters...")
    pi_c, theta_c, r_c, ll_c = multinomial_mixture_EM(counts, n_clusters=C)
    n_params = C * (K - 1) + (C - 1)
    results[C] = {'pi': pi_c, 'theta': theta_c, 'r': r_c, 'll': ll_c, 'n_params': n_params}

#%% [code]
# Method 1: BIC / AIC for WHOLE DATASET

for C in cluster_range:
    res = results[C]
    bic = -2 * res['ll'] + res['n_params'] * np.log(S)
    aic = -2 * res['ll'] + 2 * res['n_params']
    results[C]['bic'] = bic
    results[C]['aic'] = aic

bics = [results[C]['bic'] for C in cluster_range]
aics = [results[C]['aic'] for C in cluster_range]

print("BIC-optimal clusters:", list(cluster_range)[np.argmin(bics)])
print("AIC-optimal clusters:", list(cluster_range)[np.argmin(aics)])

plt.figure()
plt.plot(list(cluster_range), bics, 'o-', label='BIC')
plt.plot(list(cluster_range), aics, 's-', label='AIC')
plt.xlabel('Number of clusters')
plt.ylabel('Score')
plt.title('BIC / AIC for entire dataset (i.e. one penalty for all subjects)')
plt.legend()
plt.show()

#%% [code]
# Method 2: Bayesian model comparison (GroupBMC)

# approximate per-subject log model evidence using BIC_i = ll_i - (k/2) * log(N_i)
n_models = len(cluster_range)
lme_matrix = np.zeros((S, n_models))  # subjects x models

for idx, C in enumerate(cluster_range):
    res = results[C]
    pi_c, theta_c = res['pi'], res['theta']
    n_params = res['n_params']
    loglik_s = np.zeros((S, C))
    for j in range(C):
        loglik_s[:, j] = np.log(pi_c[j]) + (counts * np.log(theta_c[j] + 1e-12)).sum(axis=1)
    ll_per_subject = logsumexp(loglik_s, axis=1)
    N_per_subject = counts.sum(axis=1)
    lme_matrix[:, idx] = ll_per_subject - (n_params / 2) * np.log(N_per_subject)

gbmc = GroupBMC(lme_matrix.transpose())
gbmc_result = gbmc.get_result()
print("GroupBMC exceedance probabilities:", gbmc_result.protected_exceedance_probability)

plt.figure()
plt.bar(list(cluster_range), gbmc_result.protected_exceedance_probability)
plt.xlabel('Number of clusters')
plt.ylabel('Exceedance probability')
plt.title('Bayesian Model Comparison (i.e. one penalty for each subject)')
plt.show()

#%% [code]
# Method 3: Cross-validation

from sklearn.model_selection import KFold

n_folds = 5
cv_ll = {C: [] for C in cluster_range}

kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
for train_idx, test_idx in kf.split(counts):
    for C in cluster_range:
        pi_c, theta_c, _, _ = multinomial_mixture_EM(counts[train_idx], n_clusters=C)
        loglik_test = np.zeros((len(test_idx), C))
        for j in range(C):
            loglik_test[:, j] = np.log(pi_c[j]) + (counts[test_idx] * np.log(theta_c[j] + 1e-12)).sum(axis=1)
        cv_ll[C].append(logsumexp(loglik_test, axis=1).sum())

mean_cv_ll = [np.mean(cv_ll[C]) for C in cluster_range]
print("CV-optimal clusters:", list(cluster_range)[np.argmax(mean_cv_ll)])

plt.figure()
plt.errorbar(list(cluster_range), mean_cv_ll, yerr=[np.std(cv_ll[C]) for C in cluster_range], fmt='o-')
plt.xlabel('Number of clusters')
plt.ylabel('Held-out log-likelihood')
plt.title('Cross-validation')
plt.show()

#%% [code]
# Method 4: Elbow plot (log-likelihood vs. number of clusters)

lls = [results[C]['ll'] for C in cluster_range]

plt.figure()
plt.plot(list(cluster_range), lls, 'o-')
plt.xlabel('Number of clusters')
plt.ylabel('Log-likelihood')
plt.title('Elbow plot')
plt.show()

#%% [code]
# Plot the results