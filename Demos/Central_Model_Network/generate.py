import pickle

import rockfish as rf
import rockfish.actions as ra
import asyncio


def get_model(label: dict, conn: rf.Connection) -> rf.Model:
    return conn.list_models(labels=label).last()


async def generate_data(model: rf.Model,
                        conn: rf.Connection,
                        target=2000,
                        max_cycles=250) -> rf.LocalDataset:
    session_target = ra.SessionTarget(target=target, max_cycles=max_cycles)

    generate_action = pickle.load(open('recommender_output.pkl', 'rb')).actions[-1]

    builder = rf.WorkflowBuilder()
    builder.add_model(model)
    builder.add_action(generate_action, alias='gen', parents=[model, session_target])
    builder.add_action(session_target, parents=['gen'])
    builder.add_action(ra.DatasetSave(name='demo_central_syn'), parents=['gen'])
    workflow = await builder.start(conn)
    print(workflow.id())
    async for log in workflow.logs():
        print(log)
    data = await (await workflow.datasets().concat(conn)).to_local(conn)
    return data


async def generate(model_label:dict):
    conn = rf.Connection.from_config()
    location1_model = await get_model(model_label, conn)
    data = await generate_data(location1_model, conn, 1500)
    data = data.to_pandas()

    with open('synthetic_location_data.csv','w') as f:
        data.to_csv(f, index=False)

    return data


asyncio.run(generate({'location': 'location1_x'}))
