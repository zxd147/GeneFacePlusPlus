import asyncio
import datetime
import aiohttp


async def send_request(character: str, audio_name: str):
    url = 'http://192.168.0.246:8042/v1/video/generate'
    headers = {
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json',
    }
    data = {
        "uid": "1024",
        "character": character,
        "paste": "default",
        "parallel": False,
        "blocking": True,
        "audio_name": audio_name
    }

    async with aiohttp.ClientSession() as session:
        print(f"1: {datetime.datetime.now()}, {character}")
        async with session.post(url, headers=headers, json=data) as response:
            result = await response.json()  # 获取响应的 JSON 数据
            print(f"2: {datetime.datetime.now()}, {character}")
            print(f"Response for character={character}: {result}")


# 定义异步主函数来并发发送请求，并在请求之间添加延迟
async def main():
    characters_and_audio = [
        ("huang", "1725247579.wav"),
        ("li", "1aa070ea.wav"),
        ("yu", "1728722731.wav"),
        # 可以继续添加更多的任务
    ]

    # 创建任务列表
    tasks = []
    for index, (character, audio_name) in enumerate(characters_and_audio):
        # 只有在上一任务执行0.5秒后才启动新任务
        task = asyncio.create_task(send_request(character, audio_name))
        tasks.append(task)

        # 在启动任务之间添加延迟，确保每个任务之间有延迟
        # if index < len(characters_and_audio) - 1:  # 不需要在最后一个请求后延迟
        await asyncio.sleep(1)

    # 等待所有任务完成
    await asyncio.gather(*tasks)


# 执行异步任务
asyncio.run(main())








