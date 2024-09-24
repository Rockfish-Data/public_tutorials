import rockfish as rf
import rockfish.actions as ra
import pandas as pd

import asyncio


def search_model(label: dict, conn: rf.Connection) -> rf.Model:
    return conn.list_models(labels=label).last()


async def generate_data(model: rf.Model, conn: rf.Connection, target) -> pd.DataFrame:
    session_target = ra.SessionTarget(target=target, max_cycles=250)

    builder = rf.WorkflowBuilder()
    builder.add_model(model)
    builder.add_action(ra.GenerateTimeGAN(), alias='gen', parents=[model, session_target])
    builder.add_action(session_target, parents=['gen'])
    builder.add_action(ra.DatasetSave(name='demo_central_syn'), parents=['gen'])
    workflow = await builder.start(conn)
    print(workflow.id())
    async for log in workflow.logs():
        print(log)
    data = await (await workflow.datasets().concat(conn)).to_local(conn)
    return data.to_pandas()


async def generate():
    conn = rf.Connection.from_env()
    location1_model = await search_model({'location': 'location1_x'}, conn)
    data = await generate_data(location1_model, conn, 1500)

    print(data.head(10))

    return data


asyncio.run(generate())
