import asyncio
import websockets


async def websocket_client():
    uri = "ws://192.168.0.246:5465"  # WebSocket 服务端的地址和端口
    async with websockets.connect(uri) as websocket:
        # 连接成功后发送消息
        await websocket.send("Hello, WebSocket server! 你好！")

        # 循环接收服务端发送的消息
        while True:
            message = await websocket.recv()
            print(f"Received message from server: {message}")


asyncio.get_event_loop().run_until_complete(websocket_client())
