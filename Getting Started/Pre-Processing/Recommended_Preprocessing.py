# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% id="dfc66a15"
# %%capture
# %pip install -U 'rockfish[labs]' -f 'https://docs.rockfish.ai/packages/index.html'

# %% id="ca45eb9e"
import rockfish as rf
import rockfish.labs as rl

# %% id="c54ace59"
# !wget --no-clobber https://docs.rockfish.ai/tutorials/finance.csv

# %% id="81a7ed03"
dataset = rf.Dataset.from_csv("finance", "finance.csv")
x = dataset.to_pandas().iloc[:5]
x.iloc[0, 0] = None
dataset = rf.Dataset.from_pandas("finance", x)
dataset.to_pandas()

# %% [markdown] id="xnGJeBsV8_UK"
# Initiate Recommendation Engine
#

# %% id="b9da23fa"
rec = rl.Recommender.from_dataset(
    dataset,
    metadata=["customer", "age", "gender"],
    other_categorical=[
        "merchant",
        "category",
        "fraud",
    ],
)
recommends = rec.recommendations()

# %% id="c7e0713a"
print(recommends.report())

# %% [markdown] id="8DG39nCU9Ftc"
# According to the report, users can take actions to preprocess the dataset before training. In terms of our available SDK methods on preprocessing steps, you can refer to [the Pre-processing page](https://docs142.rockfish.ai/pre-processing.html) for more details.
#
