import rockfish as rf
import rockfish.labs as rl
import rockfish.actions as ra
from rockfish.labs.dataset_properties import DatasetPropertyExtractor
from rockfish.labs.recommender import ModelType
from rockfish.labs.steps import ModelSelection, Recommender
import pickle

from rockfish.labs.vis import custom_plot
from sqlalchemy.util import asyncio

from Demos.Template.train import runtime


def get_dataset(dbrx_url, table_name):
    return ra.DatabricksSqlLoad(
        token="{{ secret.databricks_token }}",
        http_path="TBD",
        server_hostname=dbrx_url,
        sql=f"SELECT * FROM rockfish_data_dev.default.{table_name}",
    )


async def run_data_quality_checks(dataset, actions, fidelity_requirements):
    conn = rf.Connection.from_config("prod")
    builder = rf.WorkflowBuilder()
    builder.add_path(dataset, *actions, ra.DatasetSave(name='onboarding-fidelity-eval'))
    workflow = await builder.start(conn)
    syn = await (await workflow.datasets().last()).to_local(conn)

    plot_config = {
        "SELECT count(fraud) AS fraud, category FROM my_table GROUP BY category": {
            "custom_plot": rl.vis.plot_kde,
            "field": "fraud"
        }
    }
    for query in fidelity_requirements[0]:
        rl.vis.custom_plot(
            datasets=[dataset, syn],
            query=query,
            plot_func=plot_config[query]["custom_plot"],
            field=plot_config[query]["fraud"]
        )



async def get_rf_recommended_workflow(
        dataset, session_key, metadata_fields,
        privacy_requirements, fidelity_requirements
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

    train_action = recommender_output.actions[0]
    train_action.config().doppelganger.epoch = 50
    train_action.config().doppelganger.epoch_checkpoint_freq = 50

    print("\nUpdating Train Model Parameters as:")
    print(f"epochs: {train_action.config().doppelganger.epoch}")

    runtime_conf = rf.WorkflowBuilder()
    runtime_conf.add_path(ra.DatastreamLoad(), *remap_actions, train_action)
    pickle.dump(runtime_conf, open('runtime_conf.pkl', 'wb'))

    pickle.dump(recommender_output.actions[-1], open('generate_conf.pkl', 'wb'))

    await run_data_quality_checks(
        dataset,
        [*list(runtime_conf.actions.values())[1:], recommender_output.actions[-1]],
        fidelity_requirements
    )

    return runtime_conf

# sample_data = get_dataset(
#     dbrx_url="dbc-224b2644-c532.cloud.databricks.com/sql/1.0/warehouses/bbdd6ab06ef5dc44/",
#     table_name="transactions_week1"
# )

sample_data = rf.Dataset.from_csv("transactions_2023-08-01_hour09", "transactions_2023-08-01_hour09.csv")

asyncio.run(
    get_rf_recommended_workflow(
        dataset=sample_data,
        session_key="customer",
        metadata_fields = ["age", "email", "gender"],
        privacy_requirements = {"email": "mask"},
        fidelity_requirements = [
            "SELECT count(fraud) AS fraud, category FROM my_table GROUP BY category",
            "SELECT avg(amount) as amount, age, gender from my_table group by age, gender",
            "SELECT avg(amount) as amount, merchant from my_table group by merchant"
        ]
    )
)