import matplotlib.pyplot as plt
import rockfish as rf
import rockfish.labs as rl
import rockfish.actions as ra
from rockfish.labs.dataset_properties import DatasetPropertyExtractor
from rockfish.labs.recommender import ModelType
from rockfish.labs.steps import ModelSelection, Recommender
import pickle
import asyncio


def get_dataset(dbrx_url, table_name):
    return ra.DatabricksSqlLoad(
        token="{{ secret.databricks_token }}",
        http_path="TBD",
        server_hostname=dbrx_url,
        sql=f"SELECT * FROM rockfish_data_dev.default.{table_name}",
    )


def data_quality_check(dataset, syn, fidelity_requirements):
    plot_configs = [
        {"custom_plot": rl.vis.plot_kde, "field": "fraud"},
        {"custom_plot": rl.vis.plot_kde, "field": "fraud"},
        {"custom_plot": rl.vis.plot_kde, "field": "fraud"},
    ]
    for query, plot_config in zip(fidelity_requirements, plot_configs):
        sns = rl.vis.custom_plot(
            datasets=[dataset, syn],
            query=query,
            plot_func=plot_config["custom_plot"],
            field=plot_config["field"],
        )
        sns.fig.suptitle(query)
        plt.show()


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
        conn = rf.Connection.from_config("prod")
        builder = rf.WorkflowBuilder()
        builder.add_path(dataset, *actions, ra.DatasetSave(name=f"{dataset.name}_syn"))
        workflow = await builder.start(conn)
        print(f"Workflow ID: {workflow.id()}")
        syn_data = await (await workflow.datasets().last()).to_local(conn)
        syn_data.to_pandas().to_csv(f"{dataset.name}_syn.csv", index=False)
    return runtime_conf

# sample_data = get_dataset(
#     dbrx_url="dbc-224b2644-c532.cloud.databricks.com/sql/1.0/warehouses/bbdd6ab06ef5dc44/",
#     table_name="transactions_week1"
# )

sample_data = rf.Dataset.from_csv("transactions_2023-08-01_hour09", "transactions_2023-08-01_hour09.csv")
fidelity_requirements = [
    "SELECT COUNT(fraud) AS fraud, category FROM my_table WHERE fraud = 1 GROUP BY category",
    "SELECT COUNT(fraud) AS fraud, age, gender FROM my_table WHERE fraud = 1 GROUP BY age, gender",
    "SELECT COUNT(fraud) AS fraud, merchant FROM my_table WHERE fraud = 1 GROUP BY merchant",
]

asyncio.run(
    get_rf_recommended_workflow(
        dataset=sample_data,
        session_key="customer",
        metadata_fields = ["age", "email", "gender"],
        privacy_requirements = {"email": "mask"},
        fidelity_requirements = fidelity_requirements
    )
)

# run the onboarding workflow to create syn data
syn = rf.Dataset.from_csv(
    "transactions_2023-08-01_hour09_syn",
    "transactions_2023-08-01_hour09_syn.csv"
)
data_quality_check(sample_data, syn, fidelity_requirements)