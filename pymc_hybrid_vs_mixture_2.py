
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

from load_data import load_data_for_experiments, load_data_for_experiment

experiment_version = "V0.3_pilot"
data = load_data_for_experiment(experiment_version)

#experiment_versions = ["V0.3_pilot", "V1.0_pilot", "V1.1_pilot"]
#data = load_data_for_experiments(experiment_versions)

counts = data['counts']


#%% [code]
# Prepare data

# SYNTHETIC DATA for model recoverability
synthetic_hybrid_counts = np.tile(np.array([10, 5, 5, 1, 1, 4]), (30, 1))

synthetic_mixture_counts = np.vstack([
    np.tile(np.array([20, 1, 1, 1, 1, 4]), (15, 1)),
    np.tile(np.array([2, 10, 10, 1, 1, 4]), (15, 1))
])

synthetic_mixture_3_counts = np.vstack([
    np.tile(np.array([20, 1, 1, 1, 1, 4]), (10, 1)),
    np.tile(np.array([2, 10, 10, 1, 1, 4]), (10, 1)),
    np.tile(np.array([1, 1, 1, 1, 20, 4]), (10, 1))
])

synthetic_hybrid_2_counts = np.tile(np.array([10, 5, 5, 1, 10, 4]), (30, 1))

synthetic_mixture_4_counts = np.vstack([
    np.tile(np.array([10, 5, 5, 1, 1, 4]), (15, 1)),
    np.tile(np.array([1, 1, 1, 1, 20, 4]), (15, 1))
])
#counts = synthetic_mixture_3_counts

S, K = counts.shape
N = counts.sum(axis=1)

#%% [code]
# Hybrid model: GPI zero + Policy reuse cued

with pm.Model() as hybrid_model:
    # positive raw weights for the tied pattern [n, m, m, e, e, 4e]
    u_e = pm.Uniform("u_e", lower=0.0, upper=1/9)
    u_n = pm.Uniform("u_n", lower=u_e, upper=1.0)
    u_m = pm.Uniform("u_m", lower=u_e, upper=1.0)

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
    loo_hybrid = az.loo(hybrid_trace)     # or az.waic(idata)
    


#%% [code]
# Mixture:
#   1) GPI zero
#   2) Policy reuse cued

with pm.Model() as mixture_model:
    # mixture weight
    w = pm.Dirichlet('w', a=np.ones(2))
    # raw weights (positive)
    u_e1  = pm.Uniform('u_e1', lower=0.0, upper=1/9)
    u_e2  = pm.Uniform('u_e2', lower=0.0, upper=1/9)
    u_n   = pm.Uniform('u_n', lower=u_e1, upper=1.0)
    u_m   = pm.Uniform('u_m', lower=u_e2, upper=1.0)

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
    loo_mixture = az.loo(mixture_trace)



#%% [code]
# Mixture:
#   1) GPI zero
#   2) Policy reuse cued
#   3) Model-based / GPI

with pm.Model() as mixture_model_3:
    # mixture weight
    w = pm.Dirichlet('w', a=np.ones(3))
    
    # raw weights (positive)
    u_e1  = pm.Uniform('u_e1', lower=0.0, upper=1/9)
    u_e2  = pm.Uniform('u_e2', lower=0.0, upper=1/9)
    u_e3  = pm.Uniform('u_e3', lower=0.0, upper=1/9)
    u_n   = pm.Uniform('u_n', lower=u_e1, upper=1.0)
    u_m   = pm.Uniform('u_m', lower=u_e2, upper=1.0)
    u_o   = pm.Uniform('u_o', lower=u_e3, upper=1.0)

    # concentration
    c1 = pm.LogNormal('c1', 0.0, 1.5)
    c2 = pm.LogNormal('c2', 0.0, 1.5)
    c3 = pm.LogNormal('c3', 0.0, 1.5)

    # component base measures
    theta1_raw = pm.math.stack([u_n, u_e1, u_e1, u_e1, u_e1, 4*u_e1])
    theta2_raw = pm.math.stack([u_e2, u_m,  u_m,  u_e2, u_e2, 4*u_e2])
    theta3_raw = pm.math.stack([u_e3, u_e3,  u_e3,  u_e3, u_o, 4*u_e3])
    theta1 = theta1_raw / pm.math.sum(theta1_raw)
    theta2 = theta2_raw / pm.math.sum(theta2_raw)
    theta3 = theta3_raw / pm.math.sum(theta3_raw)
    
    alpha1 = c1 * theta1
    alpha2 = c2 * theta2
    alpha3 = c3 * theta3

    # subject-level mixture likelihood (marginal over z)
    # PyMC has DirichletMultinomial: pm.DirichletMultinomial
    like1 = pm.DirichletMultinomial.dist(a=alpha1, n=counts.sum(axis=1))
    like2 = pm.DirichletMultinomial.dist(a=alpha2, n=counts.sum(axis=1))
    like3 = pm.DirichletMultinomial.dist(a=alpha3, n=counts.sum(axis=1))

    # mixture across subjects
    pm.Mixture('x', w, comp_dists=[like1, like2, like3], observed=counts)

    mixture_trace_3 = pm.sample(2000, tune=2000, target_accept=0.9, idata_kwargs={"log_likelihood": True})
    loo_mixture_3 = az.loo(mixture_trace_3)



#%% [code]
# Hybrid model 2: GPI zero + Policy reuse cued + MB/GPI

with pm.Model() as hybrid_model_2:
    # positive raw weights for the tied pattern [n, m, m, e, o, 4e]
    u_e = pm.Uniform("u_e", lower=0.0, upper=1/9)
    u_n = pm.Uniform("u_n", lower=u_e, upper=1.0)
    u_m = pm.Uniform("u_m", lower=u_e, upper=1.0)
    u_o = pm.Uniform("u_o", lower=u_e, upper=1.0)

    # total concentration (how similar subjects are)
    c = pm.LogNormal("c", mu=0.0, sigma=1.5)

    # construct tied base proportions
    theta_raw = at.stack([u_n, u_m, u_m, u_e, u_o, 4*u_e])
    theta = theta_raw / theta_raw.sum()

    # Dirichlet parameters
    alpha = c * theta

    # vectorized Dirichlet–Multinomial over subjects
    pm.DirichletMultinomial("x", a=alpha, n=N, shape=(S, K), observed=counts)

    hybrid_trace_2 = pm.sample(2000, tune=2000, target_accept=0.9, chains=4, idata_kwargs={"log_likelihood": True})
    # predictive accuracy (subject-level pointwise log-lik is handled internally)
    loo_hybrid_2 = az.loo(hybrid_trace_2)



#%% [code]
# Mixture 4:
#   1) GPI zero + Policy reuse cued
#   2) MB/GPI

with pm.Model() as mixture_model_4:
    # mixture weight
    w = pm.Dirichlet('w', a=np.ones(2))
    
    # raw weights (positive)
    u_e1  = pm.Uniform('u_e1', lower=0.0, upper=1/9)
    u_e2  = pm.Uniform('u_e2', lower=0.0, upper=1/9)
    u_n   = pm.Uniform('u_n', lower=u_e1, upper=1.0)
    u_m   = pm.Uniform('u_m', lower=u_e1, upper=1.0)
    u_o   = pm.Uniform('u_o', lower=u_e2, upper=1.0)

    # concentration
    c1 = pm.LogNormal('c1', 0.0, 1.5)
    c2 = pm.LogNormal('c2', 0.0, 1.5)

    # component base measures
    # Component 1: GPI zero + Policy reuse cued [n, m, m, e, e, 4e]
    theta1_raw = pm.math.stack([u_n, u_m, u_m, u_e1, u_e1, 4*u_e1])
    # Component 2: MB/GPI [e, e, e, e, o, 4e]
    theta2_raw = pm.math.stack([u_e2, u_e2, u_e2, u_e2, u_o, 4*u_e2])
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

    mixture_trace_4 = pm.sample(2000, tune=2000, target_accept=0.9, idata_kwargs={"log_likelihood": True})
    loo_mixture_4 = az.loo(mixture_trace_4)




#%% [code]
# Compare models

az.compare({
    "hybrid": hybrid_trace,
    "mixture": mixture_trace,
    "mixture_3": mixture_trace_3,
    "hybrid_2": hybrid_trace_2,
    "mixture_4": mixture_trace_4
})

# %%