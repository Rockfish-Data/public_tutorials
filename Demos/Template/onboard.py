import rockfish as rf
import rockfish.actions as ra
from rockfish.labs.dataset_properties import DatasetPropertyExtractor
from rockfish.labs.recommender import ModelType
from rockfish.labs.steps import ModelSelection, Recommender
import pickle


def get_rf_recommended_workflow(
        filepath, session_key, metadata_fields,
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

    # SAVE RUNNING WORKFLOW BUILDER (with preprocess + train actions, because this won't change per model)
    runtime_conf = rf.WorkflowBuilder()
    runtime_conf.add_path(ra.DatastreamLoad(), *remap_actions, *recommender_output.actions[:-1])
    pickle.dump(runtime_conf, open('runtime_conf.pkl', 'wb'))

    # JUST SAVE GEN ACTION (because you need to add a model as a source, and this changes per generate task)
    pickle.dump(recommender_output.actions[-1], open('generate_conf.pkl', 'wb'))

    return runtime_conf

sample_data_filepath = "test.csv"

# ONLY CHANGE THIS PER DEMO USE CASE
# e.g. for AI model training, no need for privacy_requirements
runtime_conf = get_rf_recommended_workflow(
    filepath=sample_data_filepath,
    session_key="customer",
    metadata_fields = ["age", "email", "gender"],
    privacy_requirements = {"mask": ["email"]},
    fidelity_requirements = {}
)