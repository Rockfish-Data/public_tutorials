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
import rockfish as rf
import rockfish.labs as rl

# %%
# download our example of timeseries data: finance.csv
# !wget --no-clobber https://docs.rockfish.ai/tutorials/finance.csv
# original source data
og_ts_dataset = rf.Dataset.from_csv("timeseries", "finance.csv")
og_ts_dataset.to_pandas()

# %% [markdown]
# ## Evaluate datasets based on specific conditions
#

# %% [markdown]
# #### Example 1: the specified field without null values
#

# %%
# manually create a dataset with missing values
df = og_ts_dataset.to_pandas()
df.loc[:4, "age"] = None  # introduce 5 missing values
null_dataset = rf.Dataset.from_pandas("null_sample", df)
null_dataset.to_pandas()

# %%
query = """SELECT *
    FROM my_table
    WHERE age IS NOT NULL"""
dataset = null_dataset.sync_sql(query)
dataset.to_pandas()

# %% [markdown]
# #### Example 2: Categorical Field Within Specified Categories
#

# %%
query = """SELECT *
    FROM my_table
    WHERE category IN ('fashion', 'health',
       'barsandrestaurants', 'food')"""

# Argument of datasets should be a list of one or more datasets
rl.vis.custom_plot([og_ts_dataset], query, rl.vis.plot_bar, "category")

# %% [markdown]
# #### Example 3: Select the Continuous Numerical Field Within a Specified Range
#

# %%
query = """SELECT *
    FROM my_table
    WHERE amount BETWEEN 50 AND 100"""
rl.vis.custom_plot([og_ts_dataset], query, rl.vis.plot_kde, "amount")

# %% [markdown]
# ### Evaluate datasets based on the aggregated values
#
# #### Example: count the number of unique values
#
# Below is the example to count the number of distinct
# "merchant" values and show them in histogram
#

# %%
query = """SELECT COUNT(DISTINCT merchant) AS "number of distinct values"
    FROM my_table
    GROUP BY customer, age, gender"""

# Argument of datasets should be a list of one or more datasets
rl.vis.custom_plot(
    [og_ts_dataset],
    query,
    rl.vis.plot_hist,
    "number of distinct values",
    binwidth=1,
)
