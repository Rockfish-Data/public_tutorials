from copy import deepcopy

import matplotlib.pyplot as plt
import pandas as pd
import rockfish as rf
import rockfish.actions as ra
from rockfish.labs.dataset_properties import DatasetPropertyExtractor
from rockfish.labs.recommender import ModelType
from rockfish.labs.steps import ModelSelection, Recommender
import pickle
import asyncio
import seaborn as sns


def get_dataset(dbrx_url, table_name):
    return ra.DatabricksSqlLoad(
        token="{{ secret.databricks_token }}",
        http_path="TBD",
        server_hostname=dbrx_url,
        sql=f"SELECT * FROM rockfish_data_dev.default.{table_name}",
    )


def plot_bar(dataset, syn, plot_config):
    d = dataset.to_pandas()
    s = syn.to_pandas()

    x_col_name = ",".join(plot_config["x"])
    y_col_name = plot_config["y"]

    # get x from dataset and syn
    for df in [d, s]:
        for col in plot_config["x"]:
            df[col] = df[col].astype(str)
        df[x_col_name] = df[plot_config["x"]].agg(''.join, axis=1)

    # compare only those categories (top k = 7) that exist in real
    d = d.sort_values(by=y_col_name, ascending=False)
    d = d.loc[d[x_col_name].isin(d[x_col_name][:7])]
    s = s.loc[s[x_col_name].isin(d[x_col_name])]

    x_col = d[x_col_name].to_list()
    x_col.extend(s[x_col_name].to_list())

    # get y from dataset and syn
    y_col = d[plot_config["y"]].to_list()
    y_col.extend(s[plot_config["y"]].to_list())

    # create hue col
    hue_col = [f"{dataset.name()}"] * len(d) + [f"{syn.name()}"] * len(s)

    hue_col_name = "Dataset"
    df = pd.DataFrame({
        x_col_name: x_col,
        y_col_name: y_col,
        hue_col_name: hue_col
    })

    fig, ax = plt.subplots()
    fig.set_figwidth(plot_config["figwidth"])
    sns.barplot(df, x=x_col_name, y=y_col_name, hue=hue_col_name)
    fig.suptitle(plot_config["title"])
    plt.xticks(rotation=45)
    fig.tight_layout()
    plt.savefig(f"{plot_config['title']}.png", dpi=500)

def data_quality_check(dataset, syn, fidelity_requirements):
    plot_configs = [
        {"custom_plot": None, "x": ["fraud"], "y": "fraud_count", "title": "Distribution of transaction type", "figwidth": 5},
        {"custom_plot": None, "x": ["age", "gender"], "y": "amount_perc", "title": "Percentage of amount per customer age, gender", "figwidth": 10},
    ]
    for query, plot_config in zip(fidelity_requirements, plot_configs):
        print(f"Performing check for: {query}")
        d = dataset.sync_sql(query)
        s = syn.sync_sql(query)
        plot_bar(dataset=d, syn=s, plot_config=plot_config)


async def get_rf_recommended_workflow(
        dataset, session_key, metadata_fields,
        privacy_requirements, fidelity_requirements,
        run_workflow=False
):
    dataset_properties = DatasetPropertyExtractor(
        dataset=dataset,
        session_key=session_key,
        metadata_fields=metadata_fields
    ).extract()

    recommender_output = Recommender(
        dataset_properties=dataset_properties,
        steps=[ModelSelection(model_type=ModelType.TIME_GAN)]
    ).run()

    print(recommender_output.report)

    remap_actions = []
    for col in privacy_requirements.keys():
        remap = ra.Transform(
            {"function": {"remap": ["delimiter_mask", col, {"delimiter":"@", "from_end": False}]}}
        )
        remap_actions.append(remap)

    # this was obtained after "tuning"
    train_action = recommender_output.actions[0]
    train_action.config().doppelganger.batch_size = 512
    train_action.config().doppelganger.epoch = 100
    train_action.config().doppelganger.epoch_checkpoint_freq = 100

    print("\nUpdating Train Model Parameters as:")
    print(f"batch_size: {train_action.config().doppelganger.batch_size}")
    print(f"epochs: {train_action.config().doppelganger.epoch}")

    runtime_conf = rf.WorkflowBuilder()
    runtime_conf.add_path(ra.DatastreamLoad(), *remap_actions, train_action)
    pickle.dump(runtime_conf, open('runtime_conf.pkl', 'wb'))

    pickle.dump(recommender_output.actions[-1], open('generate_conf.pkl', 'wb'))

    if run_workflow:
        actions = [*list(runtime_conf.actions.values())[1:], recommender_output.actions[-1]]
        conn = rf.Connection.from_config()
        builder = rf.WorkflowBuilder()
        builder.add_path(dataset, *actions, ra.DatasetSave(name=f"{dataset.name()}_syn"))
        workflow = await builder.start(conn)
        print(f"Workflow ID: {workflow.id()}")
        syn_data = await (await workflow.datasets().last()).to_local(conn)
        syn_data.to_pandas().to_csv(f"{dataset.name()}_syn.csv", index=False)
    return runtime_conf

# sample_data = get_dataset(
#     dbrx_url="dbc-224b2644-c532.cloud.databricks.com/sql/1.0/warehouses/bbdd6ab06ef5dc44/",
#     table_name="transactions_week1"
# )

sample_data = rf.Dataset.from_csv("Real", "transactions_2023-08-01_hour09.csv")
fidelity_requirements = [
    "SELECT fraud, COUNT(fraud) AS fraud_count FROM my_table GROUP BY fraud",
    "SELECT (SUM(amount) * 100)/SUM(SUM(amount)) OVER () as amount_perc, age, gender FROM my_table GROUP BY age, gender",
]

asyncio.run(
    get_rf_recommended_workflow(
        dataset=sample_data,
        session_key="customer",
        metadata_fields = ["age", "email", "gender"],
        privacy_requirements = {"email": "mask"},
        fidelity_requirements = fidelity_requirements,
        # run_workflow=True  # run the onboarding workflow to create syn data
    )
)

syn = rf.Dataset.from_csv(
    "Rockfish",
    "transactions_2023-08-01_hour09_syn.csv"
)
data_quality_check(sample_data, syn, fidelity_requirements)