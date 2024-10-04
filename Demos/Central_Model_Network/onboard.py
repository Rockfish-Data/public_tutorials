import asyncio

import rockfish as rf
import rockfish.actions as ra
import rockfish.labs as rl
from rockfish.labs.dataset_properties import DatasetPropertyExtractor, DatasetType
from rockfish.labs.recommender import ModelType
from rockfish.labs.steps import ModelSelection, Recommender
import pickle


async def compute_fidelity(dataset, recommender_output):
    conn = rf.Connection.from_config()
    builder = rf.WorkflowBuilder()
    builder.add_path(dataset, *recommender_output.actions, ra.DatasetSave(name='onboarding-fidelity-eval'))
    workflow = await builder.start(conn)

    syn = await (await workflow.datasets().last()).to_local(conn)

    fidelity_score = rl.metrics.marginal_dist_score(dataset, syn)
    return fidelity_score


async def get_rf_recommended_workflow(
        filepath, session_key=None, metadata_fields=None,
        privacy_requirements=None, fidelity_requirements=None,
        model_customizations=None
):
    dataset = rf.Dataset.from_csv("sample_data", filepath)
    dataset_properties = DatasetPropertyExtractor(
        dataset=dataset,
        dataset_type=DatasetType.TABULAR
    ).extract()

    recommender_output = Recommender(
        dataset_properties=dataset_properties,
        steps=[ModelSelection()]
    ).run()

    print(recommender_output.report)

    # CUSTOMIZE MODEL
    print('\nUpdating Train Model Parameters as:')
    config = recommender_output.actions[0].config()["tabular-gan"]
    for k, v in model_customizations.items():
        config[k] = v
        print(f'{k}: {v}')

    print('\nEvaluating Synthetic Data Quality:')
    fidelity_score = await compute_fidelity(dataset, recommender_output)
    print(f'Fidelity Score: {fidelity_score:.4f}')

    # SAVE RUNNING WORKFLOW BUILDER (with preprocess + train actions, because this won't change per model)
    runtime_conf = rf.WorkflowBuilder()
    runtime_conf.add_path(ra.DatastreamLoad(), *recommender_output.actions[:-1])
    pickle.dump(runtime_conf, open('runtime_conf.pkl', 'wb'))

    # JUST SAVE GEN ACTION (because you need to add a model as a source, and this changes per generate task)
    pickle.dump(recommender_output.actions[-1], open('generate_conf.pkl', 'wb'))
    return runtime_conf


sample_data_filepath = "location3_hours/location3_2023-08-06_hour00.csv"

# ONLY CHANGE THIS PER DEMO USE CASE
# e.g. for AI model training, no need for privacy_requirements
asyncio.run(
    get_rf_recommended_workflow(
        filepath=sample_data_filepath,
        privacy_requirements={},
        fidelity_requirements={},
        model_customizations={'epochs': 200}
    )
)
