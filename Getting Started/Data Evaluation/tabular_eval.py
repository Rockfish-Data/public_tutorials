# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: rockfish
#     language: python
#     name: python3
# ---

# %%
# %%capture
# %pip install -U 'rockfish[labs]' -f 'https://docs.rockfish.ai/packages/index.html'

# %%
import pandas as pd
import random

import rockfish as rf
import rockfish.labs as rl


# %%
def generate_data(num_rows, rng):
    numerical_1 = [rng.uniform(1, 100) for _ in range(num_rows)]
    numerical_2 = [rng.random() * 10 for _ in range(num_rows)]
    categorical_1 = rng.choices(
        ["A", "B", "C"], weights=[1, 1, 2], k=num_rows
    )
    categorical_2 = rng.choices(
        ["X", "Y", "Z"], weights=[1, 2, 3], k=num_rows
    )

    data = {
        "numerical_1": numerical_1,
        "numerical_2": numerical_2,
        "categorical_1": categorical_1,
        "categorical_2": categorical_2,
    }
    return pd.DataFrame(data)


rng = random.Random(42)
data = rf.Dataset.from_pandas("sample1", generate_data(100, rng))
syn = rf.Dataset.from_pandas("sample2", generate_data(100, rng))

# %% [markdown]
# ### Overall Fidelity Score
#

# %%
# get the default weighted average score on marginal distribution
rl.metrics.marginal_dist_score(dataset=data, syn=syn)

# %% [markdown]
# ### Mechanisms Across Different Perspectives
#

# %% [markdown]
# #### Individual field measurement
#

# %%
# categorical fields - field 1
rf.labs.vis.plot_bar([data, syn], "categorical_1")

# %%
# compute the similarity on the distribution of the categorical field
# TV distance ranges between 0 and 1, the lower the better
rl.metrics.tv_distance(data, syn, "categorical_1")

# %%
# categorical fields - field 2
rf.labs.vis.plot_bar([data, syn], "categorical_2")

# %%
# compute the similarity on the distribution of the categorical field
# TV distance ranges between 0 and 1, the lower the better
rl.metrics.tv_distance(data, syn, "categorical_2")

# %%
# numerical fields - field 1
rf.labs.vis.plot_kde([data, syn], "numerical_1")

# %%
# compute the similarity on the distribution of the numerical field
# KS distance ranges between 0 and 1, the lower the better
rl.metrics.ks_distance(data, syn, "numerical_1")

# %%
# numerical fields - field 2
rf.labs.vis.plot_kde([data, syn], "numerical_2")

# %%
# compute the similarity on the distribution of the numerical field
# KS distance ranges between 0 and 1, the lower the better
rl.metrics.ks_distance(data, syn, "numerical_2")

# %% [markdown]
# #### Correlation measurement
#

# %%
# correlation
rl.vis.plot_correlation([data, syn], "numerical_1", "numerical_2")

# %%
rl.vis.plot_correlation_heatmap([data, syn], ["numerical_1", "numerical_2"])

# %%
rl.metrics.correlation_score(data, syn, ["numerical_1", "numerical_2"])

# %% [markdown]
# #### Association measurement
#

# %%
rl.vis.plot_association_heatmap(
    [data, syn], ["categorical_1", "categorical_2"]
)

# %%
rl.metrics.association_score(data, syn, ["categorical_1", "categorical_2"])
