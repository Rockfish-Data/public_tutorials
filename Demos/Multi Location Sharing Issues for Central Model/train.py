import rockfish as rf
import rockfish.actions as ra

import asyncio

params = {
    'encoder': ra.TrainTimeGAN.DatasetConfig(
        timestamp=ra.TrainTimeGAN.TimestampConfig(field="timestamp"),
        metadata=[
            ra.TrainTimeGAN.FieldConfig(field="customer", type="session"),
            ra.TrainTimeGAN.FieldConfig(field="age", type="categorical"),
            ra.TrainTimeGAN.FieldConfig(field="gender", type="categorical"),
        ],
        measurements=[
            ra.TrainTimeGAN.FieldConfig(field="merchant", type="categorical"),
            ra.TrainTimeGAN.FieldConfig(field="category", type="categorical"),
            ra.TrainTimeGAN.FieldConfig(field="amount", type="continuous"),
            ra.TrainTimeGAN.FieldConfig(field="fraud", type="categorical"),
        ],
    ),
    'model': ra.TrainTimeGAN.DGConfig(
        sample_len=4,
        epoch=250,
        epoch_checkpoint_freq=100,
        sessions=4000,
        batch_size=512,
        activate_normalization_per_sample=False,
        g_lr=0.001,
        d_lr=0.001,
        attr_d_lr=0.001,
        generator_attribute_num_units=100,
        generator_attribute_num_layers=3,
        generator_feature_num_units=100,
        generator_feature_num_layers=5,
        discriminator_num_layers=5,
        discriminator_num_units=200,
        attr_discriminator_num_layers=5,
        attr_discriminator_num_units=200
    )
}


async def train():
    conn = rf.Connection.from_env()

    # from recommended actions from onboard
    encoder_config = params['encoder']

    model_config = params['model']

    config = ra.TrainTimeGAN.Config(
        encoder=encoder_config,
        doppelganger=model_config,
    )
    train_action = ra.TrainTimeGAN(config)

    builder = rf.WorkflowBuilder()
    (builder.add_action
     (ra.DatastreamLoad(), alias='stream-load'))
    builder.add_action(train_action, parents=['stream-load'])
    workflow = await builder.start(conn)
    print(workflow.id())

    async for model in workflow.models():
        print(model)


asyncio.run(train())
