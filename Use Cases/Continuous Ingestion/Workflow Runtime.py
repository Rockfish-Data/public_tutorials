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
# # Workflow Runtime: This initiates the training runtime that trains periodically ingested data and stores the models that can be used for synthesis later

# %% [markdown]
# ### Install Rockfish

# %%
# %%capture
# %pip install -U 'rockfish[labs]==0.23.0' -f 'https://docs142.rockfish.ai/packages/index.html'

# %%
import rockfish as rf
import rockfish.actions as ra

import time
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
# ### Provide the inputs to the workflow: The training actions
#
# These actions can be obtained from the onboarding process' recommender, or manually set based on the user's requirements.

# %%
with open('train_actions.pickle', 'rb') as f:
    train_actions = pickle.load(f)

# %%
stream = ra.DatastreamLoad()

builder = rf.WorkflowBuilder()
builder.add(stream, alias="input")
builder.add_path(*train_actions, parents=["input"], alias="train_actions")
workflow = await builder.start(conn)
print(f'Workflow ID: {workflow.id()}')

# %% [markdown]
#  
# ### Write the data files to the workflow stream
# - each input is a dataset
# - each output is a trained model stored to the model_store

# %% [markdown]
# ### Write data files to the workflow stream
#
# Replace the workflow ID with the actual workflow ID of the workflow that was set up

# %% [markdown]
# ### Download the sample files for the datastream workflow

# %%
# %%capture
# !wget --no-clobber https://docs142.rockfish.ai/tutorials/finance-1.csv
# !wget --no-clobber https://docs142.rockfish.ai/tutorials/finance-2.csv
# !wget --no-clobber https://docs142.rockfish.ai/tutorials/finance-3.csv
# !wget --no-clobber https://docs142.rockfish.ai/tutorials/finance-4.csv

# %%
workflow_id = 'workflow ID here'
workflow = await conn.get_workflow(workflow_id)

# %%
for file_num in range(1,4):
  data = rf.Dataset.from_csv('finance', f'finance-{file_num}.csv')
  await workflow.write_datastream("input", data)
  print(f'Writing finance-{file_num} to datastream...')
  time.sleep(10)

# %% [markdown]
#

# %% [markdown]
# ### Optional: Add custom labels to the models that are generated
#
# These labels can be used later to filter models based off custom parameters

# %%
usage = ['experimental', 'staging', 'production', 'improvement']
i = 0
async for model in (conn.list_models(labels={'workflow_id':workflow_id})):
    await model.add_labels(conn, usage=usage[i])
    i+=1
