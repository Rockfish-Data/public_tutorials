import aiohttp
import rockfish as rf
import asyncio


async def gg():
    conn = rf.Connection.from_config()
    await (await conn.get_workflow('5xrH686l8ctB5kWdQAsfXo')).stop()


asyncio.run(gg())
