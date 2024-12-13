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
#     name: python3
# ---

# %% id="JRA0zwf6EICX"
# %%capture
# %pip install -U 'rockfish[labs]' -f 'https://docs.rockfish.ai/packages/index.html'

# %% id="SS9uVPUpETyJ"
import rockfish as rf
import rockfish.actions as ra
import rockfish.labs as rl

# %% [markdown]
# Please replace `YOUR_API_KEY` with the assigned API key string. Note that it should be without quotes.
#
# For example, if the assigned API Key is `abcd1234`, you can do the following
#
# ```python
# %env ROCKFISH_API_KEY=abcd1234
# conn = rf.Connection.from_env()
# ```
#
# If you do not have API Key, please reach out to support@rockfish.ai.
#

# %% id="cAKCIort9ilu"
# %env ROCKFISH_API_KEY=YOUR_API_KEY
conn = rf.Connection.from_env()

# %% colab={"base_uri": "https://localhost:8080/"} id="7GvaKb_aEdwM" outputId="431515a9-10a7-4268-90c6-2a78d187b6b2"
# download our example of timeseries data: finance.csv
# !wget --no-clobber https://docs.rockfish.ai/tutorials/finance.csv

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} id="9WkBz993EjIM" outputId="1f7fe299-f81d-4e30-ef98-2effeaefc8ee"
dataset = rf.Dataset.from_csv("finance", "finance.csv")
dataset.to_pandas()

# %% [markdown] id="ZnCqqO5m-9-i"
# Get valid merchant-category pairs present in the train dataset:
#

# %% id="rPlpKqYn9jpw"
df = dataset.to_pandas()
merchant_to_category = {}
for mer, cat in zip(df["merchant"], df["category"]):
    valid_cats = merchant_to_category.get(mer, [])
    if cat not in valid_cats:
        valid_cats.append(cat)
    merchant_to_category[mer] = valid_cats

# %% [markdown] id="hbnPqSq3_EPo"
# These will be used to confirm that the synthetic dataset also has valid merchant-category pairs.
#

# %% [markdown] id="T57V8QMfcPsr"
# ### Join Dependent Fields
#

# %% id="AhHbqVMgEv_w"
join_fields = ra.JoinFields(fields=["merchant", "category"])

# %% [markdown] id="TTG93PYPcT8K"
# ### Train Model
#

# %% id="dFQsdRvCGpD-"
config = ra.TrainTimeGAN.Config(
    encoder=ra.TrainTimeGAN.DatasetConfig(
        timestamp=ra.TrainTimeGAN.TimestampConfig(field="timestamp"),
        metadata=[
            ra.TrainTimeGAN.FieldConfig(field="age", type="categorical"),
            ra.TrainTimeGAN.FieldConfig(field="customer", type="session"),
        ],
        measurements=[
            ra.TrainTimeGAN.FieldConfig(
                field="merchant;category", type="categorical"
            ),
            ra.TrainTimeGAN.FieldConfig(field="amount", type="continuous"),
            ra.TrainTimeGAN.FieldConfig(field="fraud", type="categorical"),
        ],
    ),
    doppelganger=ra.TrainTimeGAN.DGConfig(
        epoch=10,
        epoch_checkpoint_freq=5,
        sample_len=2,
        batch_size=1255,
    ),
)
train = ra.TrainTimeGAN(config)

# %% colab={"base_uri": "https://localhost:8080/"} id="N_WEG9m6J0m9" outputId="794a55f3-996e-48a1-ee01-77ddd8993811"
builder = rf.WorkflowBuilder()
builder.add_path(dataset, join_fields, train)
workflow = await builder.start(conn)
print(f"Workflow: {workflow.id()}")

# %% colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["d3fc97c14ec547cc89c8c3667cd498c3", "6a032143a7bc40a99dd0d30e4cf98426", "7290c770152347ac83bb58c117c5b8ec", "bd95f09eea9e4d689a743ed781ab9592", "5dc27d57eee64d539f9456f55f97156b", "3277ea36259c4cc09a3a6f0a230421d7", "7d2a1afc62cd46ea9d0d4251005b41b0", "f555b6785ab24dfebe86d1cd81879ab8", "0af1476b5d7d45f19a3abb218a1471a9", "37b7451529e54a8bb07a3834659410cf", "2c2c1499057a486292c376f87fb40c33"]} id="m38QXMBhJ9H9" outputId="1dcb27e5-812d-4c15-f2c8-a936fd5e36f7"
async for progress in workflow.progress().notebook():
    pass

# %% [markdown] id="iencLp9fcXLE"
# ### Generate Synthetic Data And Split Dependent Fields
#

# %% colab={"base_uri": "https://localhost:8080/"} id="glMocaqybuWp" outputId="e5ecacff-7e99-40a1-a2b9-8fe0f0c57d78"
model = await workflow.models().last()
model

# %% id="meRijyelIhd7"
config.doppelganger.sessions = 500
generate = ra.GenerateTimeGAN(config)
split_field = ra.SplitField(field="merchant;category")
save = ra.DatasetSave({"name": "synthetic"})

# %% colab={"base_uri": "https://localhost:8080/"} id="W1Tb4CXvbs2Z" outputId="e4448df5-53b0-41f6-ec22-b9856cc10d8d"
builder = rf.WorkflowBuilder()
builder.add_path(model, generate, split_field, save)
workflow = await builder.start(conn)
print(f"Workflow: {workflow.id()}")

# %% colab={"base_uri": "https://localhost:8080/"} id="tIoPUPdrb7IH" outputId="946e01e1-cf46-4648-d92d-4edc0d511d25"
async for log in workflow.logs():
    print(log)

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} id="07prAeuTI7Sl" outputId="c1f3c73f-f483-48c2-c86a-f8ae6e1c7541"
syn = None
async for sds in workflow.datasets():
    syn = await sds.to_local(conn)
syn.to_pandas()

# %% [markdown] id="JKWH6zSPB2p4"
# ### Evaluate Synthetic Dataset
#

# %% [markdown] id="bFyhA6NxBzEm"
# Check if synthetic dataset has valid merchant-category pairs:
#

# %% id="xX2lqs4PB7aw"
syn_df = syn.to_pandas()

# %% id="W2i64Hz-CA13"
for mer, cat in zip(syn_df["merchant"], syn_df["category"]):
    assert cat in merchant_to_category.get(mer)
