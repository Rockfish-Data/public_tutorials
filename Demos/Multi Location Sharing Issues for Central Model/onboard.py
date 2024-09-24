import rockfish as rf
import rockfish.actions as ra

import asyncio

from rockfish.labs.dataset_properties import DatasetPropertyExtractor
from rockfish.labs.recommender import ModelType
from rockfish.labs.steps import ModelSelection, Recommender


def privacy_actions(privacy_requirements):
    remap_actions = []
    for col_to_mask in privacy_requirements:
        remap = ra.Transform(
            {"function": {"remap": ["delimiter_mask", col_to_mask, {"delimiter": "@", "from_end": False}]}})
        remap_actions.append(remap)
    return remap_actions


async def onboard(file_path, privacy_requirements):
    data = rf.Dataset.from_csv('location1_onboard', file_path)
    dataset_properties = DatasetPropertyExtractor(
        data,
        session_key="customer",
        metadata_fields=["age", "gender"],
    ).extract()

    model_selection = ModelSelection(
        model_type=ModelType.TIME_GAN
    )

    privacy_parameters = privacy_actions(privacy_requirements)

    # initialize Recommender to only give the required recommendations
    recommender_output = Recommender(
        dataset_properties,
        steps=[model_selection],

    ).run()

    print(recommender_output.report)
    return recommender_output


asyncio.run(onboard('./location1.csv', 'email'))

