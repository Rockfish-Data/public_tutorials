import pandas as pd
import numpy as np
import rockfish as rf
import rockfish.actions as ra
import pyarrow as pa
import pickle
import asyncio

from main import RANSynCoders

from sklearn.preprocessing import MinMaxScaler


# https://github.com/shinan6/Longitudinal-Rockfish/blob/main/notebooks/Training_On_August%2BSynthetic_June%2BJuly.ipynb
def train_model(data, syn_data):
    x_train_og = pd.concat(data)
    x_test = pd.read_csv('test.csv')
    y_test = pd.read_csv('labels.csv', index_col=[0])

    t_train = np.tile(x_train_og.index.values.reshape(-1, 1), (1, x_train_og.shape[1]))
    t_test = np.tile(x_test.index.values.reshape(-1, 1), (1, x_train_og.shape[1]))
    xscaler = MinMaxScaler()
    x_train_scaled = xscaler.fit_transform(x_train_og.values)
    x_test_scaled = xscaler.transform(x_test.values)

    #hyperparams
    N = 5 * round((x_train_og.shape[1] / 3) / 5)  # 10 for both bootstrap sample size and number of estimators
    encoder_layers = 1  # number of hidden layers for each encoder
    decoder_layers = 2  # number of hidden layers for each decoder
    z = int((N / 2) - 1)  # size of latent space
    activation = 'relu'
    output_activation = 'sigmoid'
    S = 5  # Number of frequency components to fit to input signals
    delta = 0.05
    batch_size = 180
    freq_warmup = 5  # pre-training epochs
    sin_warmup = 5  # synchronization pre-training
    epochs = 10

    model = RANSynCoders(
        n_estimators=N,
        max_features=N,
        encoding_depth=encoder_layers,
        latent_dim=z,
        decoding_depth=decoder_layers,
        activation=activation,
        output_activation=output_activation,
        delta=delta,
        synchronize=True,
        max_freqs=S,
    )
    model.fit(x_train_scaled, t_train, epochs=epochs, batch_size=batch_size, freq_warmup=freq_warmup,
              sin_warmup=sin_warmup)
    model.save('default_model.z')

    x_train_syn = pd.concat([pd.concat(data), pd.concat(syn_data)])
    t_train = np.tile(x_train_syn.index.values.reshape(-1, 1), (1, x_train_syn.shape[1]))
    t_test = np.tile(x_test.index.values.reshape(-1, 1), (1, x_train_syn.shape[1]))
    xscaler = MinMaxScaler()
    x_train_scaled = xscaler.fit_transform(x_train_syn.values)
    x_test_scaled = xscaler.transform(x_test.values)

    # hyperparams
    N = 5 * round((x_train_syn.shape[1] / 3) / 5)  # 10 for both bootstrap sample size and number of estimators
    encoder_layers = 1  # number of hidden layers for each encoder
    decoder_layers = 2  # number of hidden layers for each decoder
    z = int((N / 2) - 1)  # size of latent space
    activation = 'relu'
    output_activation = 'sigmoid'
    S = 5  # Number of frequency components to fit to input signals
    delta = 0.05
    batch_size = 180
    freq_warmup = 5  # pre-training epochs
    sin_warmup = 5  # synchronization pre-training
    epochs = 10

    model = RANSynCoders(
        n_estimators=N,
        max_features=N,
        encoding_depth=encoder_layers,
        latent_dim=z,
        decoding_depth=decoder_layers,
        activation=activation,
        output_activation=output_activation,
        delta=delta,
        synchronize=True,
        max_freqs=S,
    )
    model.fit(x_train_scaled, t_train, epochs=epochs, batch_size=batch_size, freq_warmup=freq_warmup,
              sin_warmup=sin_warmup)
    model.save('synthetic_model.z')


async def get_synthetic_data(model_to_gen_conf):
    # connect to Rockfish platform
    conn = rf.Connection.from_config()

    syn_datasets = []
    for model_label, gen_params in model_to_gen_conf.items():
        print(f"generating from model {model_label} with params = {gen_params}")
        generate_conf = pickle.load(open("generate_conf.pkl", "rb"))

        # USUALLY IN THE DEMO WE WOULD SHOW GENERATION LIVE, SO PUT A WORKFLOW_ID HERE WITH ALREADY TRAINED MODELS
        model = await conn.list_models(labels={"kind": model_label}).last()

        builder = rf.WorkflowBuilder()
        builder.add_path(model, generate_conf, ra.DatasetSave(name="synthetic"))
        workflow = await builder.start(conn)
        syn_datasets.append((await workflow.datasets().concat(conn)).table)

    return pa.concat_tables(syn_datasets)

async def generate():
    # ONLY CHANGE THIS PER DEMO USE CASE
    # e.g. for product demo, we want to show blending and amplification
    #      for AI model training, we want to show generating missing location data
    model_label_to_gen_conf = {
        "location3.csv": {
            "sessions": 1500,
        },
        # EXAMPLE:
        # "jan": {
        #     "sessions": 500,
        # }
        # "feb": {
        #     "sessions": 500,
        # }
    }
    syn_data = await get_synthetic_data(model_label_to_gen_conf)

    # DOWNSTREAM CODE THAT USES SYN DATA GOES HERE
    # e.g. for product demo, save syn_data to file
    #      for AI model training, add xgboost/ransyncoder code here
    pa.csv.write_csv(syn_data, "location3_synthetic.csv")

    loc1_data = pd.read_csv("location1.csv")
    loc2_data = pd.read_csv("location2.csv")
    loc3_data = pd.read_csv("location3_synthetic.csv")
    train_model([loc1_data, loc2_data], [loc3_data])



asyncio.run(generate())