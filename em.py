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

def compute_cluster_loglikelihood(pi_j, theta_j, counts):
    """Compute log-likelihood for a single cluster.
    
    The probability is computed as:
    P(data | cluster j) = π_j * ∏_k (θ_j[k])^(counts[k])
    
    In log space:
    log P(data | cluster j) = log(π_j) + Σ_k counts[k] * log(θ_j[k])
    
    Args:
        pi_j: Cluster probability (scalar)
        theta_j: Cluster parameters (1D array)
        counts: Count data (2D array: subjects x categories)
    
    Returns:
        1D array of log-likelihoods for each subject
    """
    return np.log(pi_j) + (counts * np.log(theta_j + 1e-12)).sum(axis=1)

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
            loglik[:, j] = compute_cluster_loglikelihood(pi[j], theta[j], counts)
        r = np.exp(loglik - logsumexp(loglik, axis=1, keepdims=True))

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
        loglik_final[:, j] = compute_cluster_loglikelihood(pi[j], theta[j], counts)
    ll = logsumexp(loglik_final, axis=1).sum()

    return pi, theta, r, ll

#%% [code]
# Run the EM algorithm

pi, theta, r, ll = multinomial_mixture_EM(df_counts.to_numpy(), n_clusters=1)

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
# Method 1: Bayesian model comparison (GroupBMC)

def plot_model_comparison(data, metric_name, cluster_range):
    """Plot bar chart with error bars for model comparison metrics"""
    means = np.mean(data, axis=0)
    sems = np.std(data, axis=0) / np.sqrt(data.shape[0])
    
    # Create the bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(cluster_range)), means, yerr=sems, 
                   capsize=5, alpha=0.7, color='steelblue')
    
    # Customize the plot
    plt.xlabel('Number of clusters')
    plt.ylabel(metric_name)
    plt.title('Model comparison')
    plt.xticks(range(len(cluster_range)), list(cluster_range))
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(bottom=means.min() - 2*sems.max(), top=means.max() + 2*sems.max())
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.savefig(f'plots/em_clusters_bms_{metric_name}.png', dpi=150, bbox_inches='tight')

    plt.show()

# approximate per-subject log model evidence using BIC_i = ll_i - (k/2) * log(N_i)
n_models = len(cluster_range)
logliks = np.zeros((S, n_models))
bics = np.zeros((S, n_models))  # subjects x models

for idx, C in enumerate(cluster_range):
    res = results[C]
    pi_c, theta_c = res['pi'], res['theta']
    n_params = res['n_params']
    loglik_s = np.zeros((S, C))
    for j in range(C):
        loglik_s[:, j] = compute_cluster_loglikelihood(pi_c[j], theta_c[j], counts)
    logliks[:, idx] = logsumexp(loglik_s, axis=1)
    N_per_subject = counts.sum(axis=1)
    bics[:, idx] = n_params * np.log(N_per_subject) - 2 * logliks[:, idx] 

lmes = -0.5 * bics
gbmc = GroupBMC(lmes.transpose())
gbmc_result = gbmc.get_result()
print("GroupBMC exceedance probabilities:", gbmc_result.protected_exceedance_probability)


plot_model_comparison(logliks, 'Log-likelihood', cluster_range)
plot_model_comparison(bics, 'BIC', cluster_range)

#%% [code]
# Method 2: BIC / AIC for WHOLE DATASET

for idx, C in enumerate(cluster_range):
    res = results[C]
    pi_c, theta_c = res['pi'], res['theta']
    n_params = res['n_params']
    loglik_s = np.zeros((S, C))
    for j in range(C):
        loglik_s[:, j] = compute_cluster_loglikelihood(pi_c[j], theta_c[j], counts)
    ll_per_subject = logsumexp(loglik_s, axis=1)
    N_per_subject = counts.sum(axis=1)

# Also compute aggregate BIC for comparison
for C in cluster_range:
    res = results[C]
    bic = -2 * res['ll'] + res['n_params'] * np.log(S)
    aic = -2 * res['ll'] + 2 * res['n_params']
    results[C]['bic'] = bic
    results[C]['aic'] = aic

bics = [results[C]['bic'] for C in cluster_range]
aics = [results[C]['aic'] for C in cluster_range]

print("BIC-optimal clusters (aggregate):", list(cluster_range)[np.argmin(bics)])
print("BIC-optimal clusters (per-subject mean):", list(cluster_range)[np.argmin(np.mean(bics_per_subject, axis=0))])
print("AIC-optimal clusters:", list(cluster_range)[np.argmin(aics)])

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
            loglik_test[:, j] = compute_cluster_loglikelihood(pi_c[j], theta_c[j], counts[test_idx])
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