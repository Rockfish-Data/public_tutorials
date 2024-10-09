import rockfish as rf
import rockfish.actions as ra
from rockfish.labs.dataset_properties import DatasetPropertyExtractor
from rockfish.labs.recommender import ModelType
from rockfish.labs.steps import ModelSelection, Recommender
import rockfish.labs as rl
import pickle
import asyncio
import matplotlib.pyplot as plt


def data_quality_check(dataset, syn, fidelity_requirements):
    plot_configs = [
        {"custom_plot": rl.vis.plot_kde, "field": "<FIELD_NAME>", "title": "<PLOT_TITLE>"},
    ]
    for query, plot_config in zip(fidelity_requirements, plot_configs):
        sns = rl.vis.custom_plot(
            datasets=[dataset, syn],
            query=query,
            plot_func=plot_config["custom_plot"],
            field=plot_config["field"],
        )
        sns.ax.set_title(plot_config["title"])
        sns.fig.tight_layout()
        plt.savefig(f"{plot_config['title']}.png", dpi=500)


async def get_rf_recommended_workflow(
        dataset, session_key=None, metadata_fields=None,
        privacy_requirements=None, fidelity_requirements=None,
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

    # IF NEED TO CUSTOMIZE MODEL HYPERPARAMS, ADD HERE
    # print('Updating Train Model Parameters as:')

    # IF PRIVACY REQUIREMENTS EXIST, ADD REMAP ACTIONS
    # remap_actions = []

    # SAVE RUNNING WORKFLOW BUILDER (with preprocess + train actions, because this won't change per model)
    runtime_conf = rf.WorkflowBuilder()
    runtime_conf.add_path(ra.DatastreamLoad(), *recommender_output.actions[:-1])
    pickle.dump(runtime_conf, open('runtime_conf.pkl', 'wb'))

    # JUST SAVE GEN ACTION (because you need to add a model as a source, and this changes per generate task)
    pickle.dump(recommender_output.actions[-1], open('generate_conf.pkl', 'wb'))

    if run_workflow:
        actions = [*list(runtime_conf.actions.values())[1:], recommender_output.actions[-1]]
        conn = rf.Connection.from_config()
        builder = rf.WorkflowBuilder()
        builder.add_path(dataset, *actions, ra.DatasetSave(name=f"{dataset.name}_syn"))
        workflow = await builder.start(conn)
        print(f"Workflow ID: {workflow.id()}")
        syn_data = await (await workflow.datasets().last()).to_local(conn)
        syn_data.to_pandas().to_csv(f"{dataset.name}_syn.csv", index=False)
    return runtime_conf


sample_data = rf.Dataset.from_csv("Real", "test.csv")
fidelity_requirements = []

# ONLY CHANGE THIS PER DEMO USE CASE
# e.g. for AI model training, no need for privacy_requirements
asyncio.run(get_rf_recommended_workflow(
    dataset=sample_data,
    session_key="customer",
    metadata_fields=["age", "email", "gender"],
    privacy_requirements={"email": "mask"},
    fidelity_requirements=fidelity_requirements,  # add SQL queries according to dataset
    # run_workflow=True  # run the onboarding workflow to create syn data
))

syn = rf.Dataset.from_csv(
    "Rockfish",
    "<PATH>"
)
data_quality_check(sample_data, syn, fidelity_requirements)