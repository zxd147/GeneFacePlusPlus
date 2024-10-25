import asyncio
import socket
import websockets
from websockets import exceptions

gWsk_self = None
gWsk_remote = None


# 处理 WebSocket 连接的函数
async def websocket_handler(websocket, path):
    global gWsk_self, gWsk_remote
    print(f"New connection from {websocket.remote_address} path {path}")

    if "192.168.0.246" in websocket.remote_address[0]:
        gWsk_self = websocket
        print("Connected: 本机")
    else:
        gWsk_remote = websocket
        print("Connected: 远程")

    try:
        async for message in websocket:
            print(f"Received message: {message}")

            if "192.168.0.246" in websocket.remote_address[0] and gWsk_remote:
                await gWsk_remote.send("我是本机，转给远程的你")
            elif gWsk_self:
                await gWsk_self.send("我是远程，转给本机的你")
            await websocket.send(message)  # 原样返回消息
    except websockets.exceptions.ConnectionClosedError:
        print(f"Connection to {websocket.remote_address} closed")
        if "192.168.0.246" in websocket.remote_address[0]:
            gWsk_self = None
            print("本地关闭")
        else:
            gWsk_remote = None
            print("远程关闭")


# 启动 WebSocket 服务端
async def main():
    server = await websockets.serve(websocket_handler, "0.0.0.0", 5465)
    print("WebSocket server started. Listening on ws://0.0.0.0:5465")
    await server.wait_closed()  # 保持服务器运行，直到关闭


# 运行事件循环
if __name__ == "__main__":
    asyncio.run(main())
