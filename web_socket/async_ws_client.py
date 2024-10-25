import asyncio
import json

import websockets
from websockets import exceptions


# 发送消息的任务，发送100个数据
async def send_messages(websocket):
    for i in range(100):
        message = json.dumps({"messages": f'{i}', "code": 123})  # 确保发送有效的 JSON
        await websocket.send(message)
        print(f"Sent: {message}")
        await asyncio.sleep(0.1)  # 模拟延迟
    print("Finished sending 100 messages.")


# 接收消息的任务，永久接收服务端的响应
async def receive_messages(websocket):
    while True:
        try:
            response = await websocket.recv()  # 接收消息
            print(f"Received: {response}")
        except websockets.exceptions.ConnectionClosedError:
            print("Connection closed by server.")
            break


# 主函数，分别启动发送和接收任务
async def client():
    uri = "ws://localhost:6543"
    async with websockets.connect(uri) as websocket_client:
        # 创建两个任务：一个负责发送消息，一个负责接收消息
        # send_task 和 receive_task 是通过 asyncio.create_task() 并发执行的。
        send_task = asyncio.create_task(send_messages(websocket_client))
        receive_task = asyncio.create_task(receive_messages(websocket_client))
        # 等待发送任务完成，接收任务会永久运行直到连接关闭
        # 显式地等待它们的完成。
        await send_task
        await receive_task  # 如果需要在接收任务完成后继续其他操作，可以选择取消 receive_task


# 启动客户端
if __name__ == "__main__":
    asyncio.run(client())
