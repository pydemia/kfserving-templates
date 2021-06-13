import aiohttp
import asyncio
import time

start_time = time.time()


async def get_pokemon(session, url):
    async with session.post(url) as resp:
        r = await resp
        return r


async def main():

    async with aiohttp.ClientSession() as session:

        tasks = []
        for number in range(1, 151):
            url = f'http://localhost:28080/v2/models/model/infer'
            tasks.append(asyncio.ensure_future(get_pokemon(session, url)))

        original_pokemon = await asyncio.gather(*tasks)

asyncio.run(main())
print("--- %s seconds ---" % (time.time() - start_time))
