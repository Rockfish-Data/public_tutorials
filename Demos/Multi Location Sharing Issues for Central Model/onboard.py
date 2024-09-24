import pickle

import rockfish as rf
import rockfish.actions as ra

import asyncio
from rockfish.labs.dataset_properties import DatasetPropertyExtractor
from rockfish.labs.recommender import ModelType
from rockfish.labs.steps import ModelSelection, Recommender


def get_rf_recommended_workflow(dataset, session_key, metadata_fields, privacy_requirements):
    dataset_properties = DatasetPropertyExtractor(dataset=dataset, session_key=session_key,
                                                  metadata_fields=metadata_fields).extract()
    recommender_output = Recommender(dataset_properties=dataset_properties,
                                     steps=[ModelSelection(model_type=ModelType.TIME_GAN)]).run()

    remap_actions = []
    for col_to_mask in privacy_requirements:
        remap = ra.Transform(
            {"function": {"remap": ["delimiter_mask", col_to_mask, {"delimiter": "@", "from_end": False}]}})
        remap_actions.append(remap)

    # action updates per use case
    recommender_output.actions = remap_actions[0] + recommender_output.actions
    return recommender_output



    # train_wb = rf.WorkflowBuilder()
    # train_wb.add_path(dataset, remap_actions[0], train_action)
    # return train_wb


async def onboard(file_path, privacy_requirements):
    data = rf.Dataset.from_csv('onboarding_data', file_path)
    recommender_output = get_rf_recommended_workflow(
        data,
        session_key="customer",
        metadata_fields=["email", "age", "gender"],
        privacy_requirements=privacy_requirements
    )
    print(recommender_output.report)
    pickle.dump(recommender_output, open('recommender_output.pkl', 'wb'))
    return recommender_output


asyncio.run(onboard('./location1.csv', 'email'))
