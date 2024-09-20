# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import rockfish as rf
import rockfish.actions as ra
from rockfish.labs.dataset_properties import DatasetPropertyExtractor
from rockfish.labs.steps import Recommender, ModelSelection
from rockfish.labs.recommender import ModelType

# %% [markdown]
# ### Load Real Data

# %%
dataset = rf.Dataset.from_csv("finance", './finance.csv')

# %%
dataset.to_pandas()

# %% [markdown]
# ### View Real Data in Dashboard
#
# Link: [TBD]

# %% [markdown]
# ### Goal: Create Synthetic Data with masked emails and more fraudulent transactions

# %% [markdown]
# #### Details

# %% jupyter={"source_hidden": true}
conn = rf.Connection.remote(
    'https://sunset-beach.rockfish.ai',
    'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE3MTE2NDc0MDQsImlzcyI6ImFwaSIsIm5iZiI6MTcxMTY0NzQwNCwidG9rZW5faWQiOiIxd3pBUWliNjRVb0c2MWVUazQ4SzBMIiwidXNlcl9pZCI6IjQ2MVNUOXZ4a0hYekpYRnJKYm4yWm0ifQ.MxG4VB5IrXQ2U_2ePUaoEN7gfy2fqPhD5tzSYYhnn2k'
)


# %% jupyter={"source_hidden": true}
def get_rf_recommended_workflow(dataset, session_key, metadata_fields, privacy_requirements):
    dataset_properties = DatasetPropertyExtractor(dataset=dataset, session_key=session_key, metadata_fields=metadata_fields).extract()
    recommender_output = Recommender(dataset_properties=dataset_properties, steps=[ModelSelection(model_type=ModelType.TIME_GAN)]).run()
    train_action = recommender_output.actions[0]

    remap_actions = []
    for col_to_mask in privacy_requirements:
        remap = ra.Transform({"function": {"remap": ["fixed_mask", col_to_mask, {"mask_length": 8, "from_end": False}]}})

    train_wb = rf.WorkflowBuilder()
    train_wb.add_path(dataset, *remap_actions, train_action)
    return train_wb


# %% jupyter={"source_hidden": true}
async def get_story_data(model_id, n_sessions, story_requirements):
    dataset_properties = DatasetPropertyExtractor(dataset=dataset, session_key="customer", metadata_fields=["email", "age", "gender"]).extract()
    recommender_output = Recommender(dataset_properties=dataset_properties, steps=[ModelSelection(model_type=ModelType.TIME_GAN)]).run()
    generate = recommender_output.actions[1]

    model = rf.Model(model_id)

    post_amplify = ra.PostAmplify({
        "query_ast": {
            "eq": ["fraud", 1],
        },
        "drop_match_percentage": 0.0,
        "drop_other_percentage": 0.5,
    })
    session_target = ra.SessionTarget(target=n_sessions, max_cycles=100, use_match_count=True)
    save = ra.DatasetSave(name="synthetic", concat_session_key="session_key")
    
    builder = rf.WorkflowBuilder()
    builder.add_model(model)
    builder.add_action(generate, parents=[model, session_target])
    builder.add_action(post_amplify, parents=[generate])
    builder.add_action(session_target, parents=[post_amplify])
    builder.add_action(save, parents=[post_amplify])

    workflow = await builder.start(conn)
    
    syn_data = await workflow.datasets().concat(conn)
    return syn_data

# %% [markdown]
# #### Create Rockfish Model From Recommended Workflow

# %%
train_wb = get_rf_recommended_workflow(
    dataset, session_key="customer", metadata_fields=["email", "age", "gender"], privacy_requirements=["email"]
)
train_workflow = await train_wb.start(conn)

# %%
model_id = (await train_workflow.models().nth(0)).id

# %% [markdown]
# #### Generate Synthetic Data Using Rockfish Model

# %%
synthetic_dataset = await get_story_data(model_id, n_sessions=5000, story_requirements=["amplify_fraud"])

# %%
synthetic_dataset.to_pandas()

# %%
synthetic_dataset.to_pandas().to_csv("finance_synthetic.csv", index=False)

# %% [markdown]
# ### View Synthetic Data in Dashboard
#
# Link: [TBD]
