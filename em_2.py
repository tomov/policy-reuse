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

from load_data import load_data

data = load_data()
counts = data['counts']

#%% [code]
# EM algorithm for two multinomial distributions


def log_multinom_component(c, p, kind):
    """Log-likelihood of one component for a single subject"""
    if kind == "H7":
        e = (1 - p)/8
        probs = np.array([p, e, e, e, e, 4*e])
    else:
        assert kind == "H4"
        e = (1 - p)/8
        probs = np.array([e, p/2, p/2, e, e, 4*e])
    return np.sum(c * np.log(probs + 1e-12))

def em_two_multinom(counts, max_iter=200, tol=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    S, K = counts.shape
    N = counts.sum(axis=1)

    # Initialize parameters
    w = 0.5
    p = 0.4  # H7 probability of choosing GPI zero
    q = 0.4  # H4 probability of choosing policy reuse cued
    loglik_prev = -np.inf

    for it in range(max_iter):
        # E-step
        logp1 = np.array([log_multinom_component(c, p, "H7") for c in counts])
        logp2 = np.array([log_multinom_component(c, q, "H4") for c in counts])
        logw1 = np.log(w + 1e-12) + logp1
        logw2 = np.log(1 - w + 1e-12) + logp2
        m = np.maximum(logw1, logw2)
        w1 = np.exp(logw1 - m)
        w2 = np.exp(logw2 - m)
        r1 = w1 / (w1 + w2)
        r2 = 1 - r1

        # M-step
        w = r1.mean()
        # weighted sufficient statistics
        A1 = np.sum(r1 * (counts[:,0] + counts[:,5]))
        B1 = np.sum(r1 * (counts[:,1] + counts[:,2] + counts[:,3] + counts[:,4]))
        p = A1 / (2*(A1 + B1))
        A2 = np.sum(r2 * counts[:,1])
        B2 = np.sum(r2 * (counts[:,0] + counts[:,2] + counts[:,3] + counts[:,4] + counts[:,5]))
        q = A2 / (A2 + B2)

        # log-likelihood
        ll = np.sum(np.log(np.exp(logw1) + np.exp(logw2)))
        if np.abs(ll - loglik_prev) < tol:
            break
        loglik_prev = ll

    # Calculate BIC
    k = 3  # number of free parameters: w, p, q
    n = S  # number of subjects
    bic = k * np.log(n) - 2 * ll

    return dict(w=w, p=p, q=q, responsibilities=np.vstack([r1,r2]).T, loglik=ll, bic=bic)

#%% [code]
# Run the EM algorithm

fit = em_two_multinom(counts)
print("EM Fit Results:")
print(f"w (mixture weight) = {fit['w']:.6f}")
print(f"p (H7 probability GPI zero) = {fit['p']:.6f}")
print(f"q (H4 probability policy reuse cued) = {fit['q']:.6f}")
print(f"Log-likelihood = {fit['loglik']:.6f}")
print(f"BIC = {fit['bic']:.6f}")
print("Responsibilities (first 10 rows):")
print(fit['responsibilities'][:10])

# %%
