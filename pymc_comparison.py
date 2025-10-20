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


#%% [code]
# Define model & run you inference


# counts: shape (S, 6)
# rows are subjects, columns are c1..c6
counts = df_counts.to_numpy()
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
