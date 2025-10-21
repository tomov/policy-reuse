#%% [markdown]
# ## Preliminaries


#%% [code]
# Includes


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import numpy as np
import arviz as az
import pytensor.tensor as at



np.set_printoptions(suppress=True,precision=3)


#%% [code]
# Load data

from load_data import load_data

data = load_data()
counts = data['counts']


#%% [code]
# Prepare data

# SYNTHETIC DATA for model recoverability
synthetic_hybrid_counts = np.tile(np.array([10, 5, 5, 1, 1, 4]), (30, 1))
synthetic_mixture_counts = np.vstack([
    np.tile(np.array([20, 1, 1, 1, 1, 4]), (15, 1)),
    np.tile(np.array([2, 10, 10, 1, 1, 4]), (15, 1))
])
#counts = synthetic_mixture_counts

S, K = counts.shape
N = counts.sum(axis=1)


#%% [code]
# Define hybrid model & run inference


with pm.Model() as hybrid_model:
    # positive raw weights for the tied pattern [n, m, m, e, e, 4e]
    u_n = pm.HalfNormal("u_n", sigma=1.0)
    u_m = pm.HalfNormal("u_m", sigma=1.0)
    u_e = pm.HalfNormal("u_e", sigma=1.0)

    # total concentration (how similar subjects are)
    c = pm.LogNormal("c", mu=0.0, sigma=1.5)

    # construct tied base proportions
    theta_raw = at.stack([u_n, u_m, u_m, u_e, u_e, 4*u_e])
    theta = theta_raw / theta_raw.sum()

    # Dirichlet parameters
    alpha = c * theta

    # vectorized Dirichlet–Multinomial over subjects
    pm.DirichletMultinomial("x", a=alpha, n=N, shape=(S, K), observed=counts)

    hybrid_trace = pm.sample(2000, tune=2000, target_accept=0.9, chains=4, idata_kwargs={"log_likelihood": True})
    # predictive accuracy (subject-level pointwise log-lik is handled internally)
    loo = az.loo(hybrid_trace)     # or az.waic(idata)
    


#%% [code]
# Define mixture model & run inference

with pm.Model() as mixture_model:
    # mixture weight
    w = pm.Dirichlet('w', a=np.ones(2))
    # raw weights (positive)
    u_n   = pm.HalfNormal('u_n', 1.0)
    u_m   = pm.HalfNormal('u_m', 1.0)
    u_e1  = pm.HalfNormal('u_e1', 1.0)
    u_e2  = pm.HalfNormal('u_e2', 1.0)

    # concentration
    c1 = pm.LogNormal('c1', 0.0, 1.5)
    c2 = pm.LogNormal('c2', 0.0, 1.5)

    # component base measures
    theta1_raw = pm.math.stack([u_n, u_e1, u_e1, u_e1, u_e1, 4*u_e1])
    theta2_raw = pm.math.stack([u_e2, u_m,  u_m,  u_e2, u_e2, 4*u_e2])
    theta1 = theta1_raw / pm.math.sum(theta1_raw)
    theta2 = theta2_raw / pm.math.sum(theta2_raw)

    alpha1 = c1 * theta1
    alpha2 = c2 * theta2

    # subject-level mixture likelihood (marginal over z)
    # PyMC has DirichletMultinomial: pm.DirichletMultinomial
    like1 = pm.DirichletMultinomial.dist(a=alpha1, n=counts.sum(axis=1))
    like2 = pm.DirichletMultinomial.dist(a=alpha2, n=counts.sum(axis=1))

    # mixture across subjects
    pm.Mixture('x', w, comp_dists=[like1, like2], observed=counts)

    mixture_trace = pm.sample(2000, tune=2000, target_accept=0.9, idata_kwargs={"log_likelihood": True})
    loo = az.loo(mixture_trace)




#%% [code]
# Compare models

az.compare({
    "hybrid": hybrid_trace,
    "mixture": mixture_trace
})

# %%
