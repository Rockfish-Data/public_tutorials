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

# %% [markdown]
#  
# ## Synthesis Runtime: This generates the synthetic data from a model provided to it

# %% [markdown]
# ### Install Rockfish

# %%
# %%capture
# %pip install -U 'rockfish[labs]==0.23.0' -f 'https://docs142.rockfish.ai/packages/index.html'

# %%
import rockfish as rf
import rockfish.actions as ra

import pickle

# %% [markdown]
# ### Connect to the Rockfish Platform
#
# ❗❗ Replace API_KEY.

# %%
API_KEY = 'insert your API key here'

conn = rf.Connection.remote('https://api.rockfish.ai', API_KEY)

# %% [markdown]
#  
# ### Provide query params to the model_store search to get appropriate models as response

# %%
async for model in conn.list_models(labels={'usage': 'production'}):
    print(model)

# %% [markdown]
#  
# ### Select a model from the list of queried models and fetch it from remote

# %%
model = await rf.Model.from_id(conn, 'model id here of the filtered model after querying')
print(model)

# %%
with open('generate_actions.pickle', 'rb') as f:
    generate_actions = pickle.load(f)

# %% [markdown]
#  
# ### Provide the model and the synthesis config to a workflow to generate a synthetic dataset as the output

# %%
builder = rf.WorkflowBuilder()
builder.add(model)
builder.add(*generate_actions, parents=[model], alias='gen')
builder.add(ra.DatasetSave(name='syn_data'), parents=['gen'])
workflow = await builder.start(conn)
print(f'Workflow ID: {workflow.id()}')

# %%
syn_data = await workflow.datasets().concat(conn)

# %%
syn_data.to_pandas()
