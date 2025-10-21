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


np.set_printoptions(suppress=True,precision=3)


#%% [code]
# Load data

from load_data import load_data

data = load_data()
counts = data['counts']


#%% [code]
# Define model & run inference

S, K = counts.shape

with pm.Model() as model:
    # Hyperpriors
    mu = pm.Dirichlet("mu", a=np.ones(K))        # population mean on simplex
    kappa = pm.Gamma("kappa", alpha=2., beta=1.) # concentration (>0)

    alpha = pm.Deterministic("alpha", kappa * mu)  # Dirichlet params

    # Likelihood (subject-specific)
    pm.DirichletMultinomial(
        "y",
        a=alpha,
        n=counts.sum(axis=1),   # total trials per subject
        observed=counts
    )

    trace = pm.sample(tune=1000, draws=2000, target_accept=0.9, chains=4)
    

#%% [code]
# Viz results

# Posterior questions
import arviz as az
az.summary(trace, var_names=["mu","kappa"])

# Get posterior means for mu (group-level probabilities)
posterior_means = trace.posterior["mu"].mean(dim=["chain", "draw"])
print("Group posterior means:", posterior_means.values)

# Get the subject-specific posterior samples
# The DirichletMultinomial gives you subject-specific choice probabilities
subject_samples = trace.posterior["y"]  # This contains the subject-specific parameters
subject_means = subject_samples.mean(dim=["chain", "draw"])
print("Subject-specific posterior means:", subject_means)

# P(mu_2 > 1/6)
p_mu2_gt = (trace.posterior["mu"].sel(mu_dim_0=1).values > 1/6).mean()
print("Pr(mu2 > 1/6 | data) =", p_mu2_gt)

# For a random subject: Pr(p2 > 1/6)
# p2 | mu,kappa ~ Beta(kappa*mu2, kappa*(1-mu2))
from scipy.stats import beta
mu2 = trace.posterior["mu"].sel(mu_dim_0=1).values.ravel()
kap = trace.posterior["kappa"].values.ravel()
p_rand_gt = (1 - beta.cdf(1/6, kap*mu2, kap*(1-mu2))).mean()
print("Pr(p2 > 1/6 for a random subject) =", p_rand_gt)

# %%
