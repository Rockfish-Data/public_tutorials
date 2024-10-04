import rockfish as rf
import rockfish.actions as ra
from rockfish.labs.dataset_properties import DatasetPropertyExtractor
from rockfish.labs.recommender import ModelType
from rockfish.labs.steps import ModelSelection, Recommender
import pickle


def get_dataset(dbrx_url, table_name):
    return ra.DatabricksSqlLoad(
        token="{{ secret.databricks_token }}",
        http_path="TBD",
        server_hostname=dbrx_url,
        sql=f"SELECT * FROM rockfish_data_dev.default.{table_name}",
    )

def get_rf_recommended_workflow(
        dataset, session_key, metadata_fields,
        privacy_requirements, fidelity_requirements
):
    dataset = rf.Dataset.from_csv("sample_data", filepath)
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
    for col_to_mask in privacy_requirements["mask"]:
        remap = ra.Transform(
            {"function": {"remap": ["fixed_mask", col_to_mask, {"mask_length": 8, "from_end": False}]}}
        )
        remap_actions.append(remap)

    runtime_conf = rf.WorkflowBuilder()
    runtime_conf.add_path(ra.DatastreamLoad(), *remap_actions, *recommender_output.actions[:-1])
    pickle.dump(runtime_conf, open('runtime_conf.pkl', 'wb'))

    pickle.dump(recommender_output.actions[-1], open('generate_conf.pkl', 'wb'))

    return runtime_conf

sample_data = get_dataset(
    dbrx_url="dbc-224b2644-c532.cloud.databricks.com/sql/1.0/warehouses/bbdd6ab06ef5dc44/",
    table_name="transactions_week1"
)

runtime_conf = get_rf_recommended_workflow(
    dataset=sample_data,
    session_key="customer",
    metadata_fields = ["age", "email", "gender"],
    privacy_requirements = {"mask": ["email"]},
    fidelity_requirements = {"amount BETWEEN 0.0 AND 500.0"}
)