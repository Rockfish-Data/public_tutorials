import asyncio

import rockfish as rf
import rockfish.actions as ra
import rockfish.labs as rl
from rockfish.labs.dataset_properties import DatasetPropertyExtractor, DatasetType
from rockfish.labs.steps import ModelSelection, Recommender
import pickle
import matplotlib.pyplot as plt


async def compute_fidelity(dataset, recommender_output):
    # This is the code that will be responsible for the generation of a sample synthetic data, however, it is commented out to allow for exact reproducibility of the demo

    # conn = rf.Connection.from_config()
    # builder = rf.WorkflowBuilder()
    # builder.add_path(dataset, *recommender_output.actions, ra.DatasetSave(name='onboarding-fidelity-eval'))
    # workflow = await builder.start(conn)
    #
    # syn_data = await (await workflow.datasets().last()).to_local(conn)

    feature = "feature_9"

    source_data = rf.Dataset.from_csv("Ideal",
                                      "datafiles/location3_hours/location3_2023-08-06_hour01.csv")
    syn_filepath = "datafiles/sample_syn.csv"
    syn_data = rf.Dataset.from_csv("Rockfish", syn_filepath)

    syn_naive_data = rf.Dataset.from_csv("Naive", "datafiles/naive_syn_data.csv")
    syn_naive_data.table = syn_naive_data.table.slice(offset=1000, length=60)

    # one of the many plotting options available in rockfish.labs
    sns = rl.vis.plot_kde([source_data, syn_naive_data, syn_data], feature, palette=['g', 'orange', 'b'])
    sns.set_xlabels("Normalized Feature9")
    plt.show()


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
    await compute_fidelity(dataset, recommender_output)

    # SAVE RUNNING WORKFLOW BUILDER (with preprocess + train actions, because this won't change per model)
    runtime_conf = rf.WorkflowBuilder()
    runtime_conf.add_path(ra.DatastreamLoad(), *recommender_output.actions[:-1])
    pickle.dump(runtime_conf, open('runtime_conf.pkl', 'wb'))

    # JUST SAVE GEN ACTION (because you need to add a model as a source, and this changes per generate task)
    pickle.dump(recommender_output.actions[-1], open('generate_conf.pkl', 'wb'))
    return runtime_conf


sample_data_filepath = "datafiles/location3_hours/location3_2023-08-06_hour01.csv"

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
