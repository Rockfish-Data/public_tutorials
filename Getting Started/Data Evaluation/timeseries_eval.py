# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: colab
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
def generate_timeseries_data(num_rows, rng):
    metadata = rng.choices(["A", "B", "C"], weights=[1, 1, 2], k=num_rows)
    timestamp = pd.date_range(start="1/1/2020", periods=num_rows, freq="D")
    numerical_1 = [rng.uniform(1, 100) for _ in range(num_rows)]
    numerical_2 = [rng.random() * 10 for _ in range(num_rows)]
    categorical_1 = rng.choices([1, 2, 3], weights=[1, 1, 2], k=num_rows)
    categorical_2 = rng.choices(
        ["X", "Y", "Z"], weights=[1, 2, 3], k=num_rows
    )

    data = {
        "metadata_field": metadata,
        "timestamp_field": timestamp,
        "numerical_1": numerical_1,
        "numerical_2": numerical_2,
        "categorical_1": categorical_1,
        "categorical_2": categorical_2,
    }

    return pd.DataFrame(data)


rng = random.Random(42)
ts_data = rf.Dataset.from_pandas(
    "sample1", generate_timeseries_data(100, rng)
)
ts_syn = rf.Dataset.from_pandas("sample2", generate_timeseries_data(100, rng))

# %% [markdown]
# ### Overall Fidelity Score
#

# %%
# get the default weighted average score on marginal distribution
# other_categorical is the list of categorical fields with numeric values
rl.metrics.marginal_dist_score(
    dataset=ts_data,
    syn=ts_syn,
    metadata=["metadata_field"],
    other_categorical=["categorical_1"],
)

# %% [markdown]
# ### Mechanisms Across Different Perspectives
#

# %% [markdown]
# #### Session length measurement
#

# %%
# compute session_length
source_sess = rf.metrics.session_length(ts_data)
syn_sess = rf.metrics.session_length(ts_syn)
# "session_length" is a fixed name from the computed datasets
rl.vis.plot_kde([source_sess, syn_sess], "session_length")

# %%
# compute the similarity on the distribution of session length
# KS distance ranges between 0 and 1, the lower the better
rl.metrics.ks_distance(source_sess, syn_sess, "session_length")

# %% [markdown]
# #### Interarrival time measurement
#

# %%
# compute interarrival time
timestamp_field = "timestamp_field"
source_interarrival = rf.metrics.interarrivals(ts_data, timestamp_field)
syn_interarrival = rf.metrics.interarrivals(ts_syn, timestamp_field)
# "interarrival" is a fixed name from the computed datasets
rf.labs.vis.plot_kde(
    [source_interarrival, syn_interarrival], "interarrival", duration_unit="s"
)

# %%
# compute the similarity on the distribution of interarrival time
# KS distance ranges between 0 and 1, the lower the better
rl.metrics.ks_distance(source_interarrival, syn_interarrival, "interarrival")

# %% [markdown]
# #### Transitions measurement
#

# %%
field_name = "categorical_1"
transitions_source = rf.metrics.transitions_within_sessions(
    ts_data, field=field_name
)
transitions_syn = rf.metrics.transitions_within_sessions(
    ts_syn, field=field_name
)
rl.vis.plot_bar(
    [transitions_source, transitions_syn],
    f"{field_name}_transitions",
    orient="horizontal",
)

# %%
# compute the similarity on the distribution of categorical field
# TV distance ranges between 0 and 1, the lower the better
rl.metrics.tv_distance(
    transitions_source, transitions_syn, f"{field_name}_transitions"
)

# %% [markdown]
# #### Individual field measurement
#

# %%
# categorical fields - field 1
rf.labs.vis.plot_bar([ts_data, ts_syn], "categorical_1")

# %%
# compute the similarity on the distribution of the categorical field
# TV distance ranges between 0 and 1, the lower the better
rl.metrics.tv_distance(ts_data, ts_syn, "categorical_1")

# %%
# categorical fields - field 2
rf.labs.vis.plot_bar([ts_data, ts_syn], "categorical_2")

# %%
# compute the similarity on the distribution of the categorical field
# TV distance ranges between 0 and 1, the lower the better
rl.metrics.tv_distance(ts_data, ts_syn, "categorical_2")

# %%
# numerical fields - field 1
rf.labs.vis.plot_kde([ts_data, ts_syn], "numerical_1")

# %%
# compute the similarity on the distribution of the numerical field
# KS distance ranges between 0 and 1, the lower the better
rl.metrics.ks_distance(ts_data, ts_syn, "numerical_1")

# %%
# numerical fields - field 2
rf.labs.vis.plot_kde([ts_data, ts_syn], "numerical_2")

# %%
# compute the similarity on the distribution of the numerical field
# KS distance ranges between 0 and 1, the lower the better
rl.metrics.ks_distance(ts_data, ts_syn, "numerical_2")

# %% [markdown]
# #### Correlation measurement
#

# %%
# correlation
rl.vis.plot_correlation([ts_data, ts_syn], "numerical_1", "numerical_2")

# %%
rl.vis.plot_correlation_heatmap(
    [ts_data, ts_syn], ["numerical_1", "numerical_2"]
)

# %%
rl.metrics.correlation_score(ts_data, ts_syn, ["numerical_1", "numerical_2"])

# %% [markdown]
# #### Association measurement
#

# %%
rl.vis.plot_association_heatmap(
    [ts_data, ts_syn], ["categorical_1", "categorical_2"]
)

# %%
rl.metrics.association_score(
    ts_data, ts_syn, ["categorical_1", "categorical_2"]
)
