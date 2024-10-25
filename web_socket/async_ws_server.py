import asyncio
from typing import Optional, Union
import websockets
from pydantic import BaseModel
from websockets import exceptions


class GenerateRequest(BaseModel):
    code: int
    sno: Optional[Union[int, str]] = '123'
    messages: Optional[str]
    audio_path: Optional[str] = None
    audio_base64: Optional[str] = None


async def process_message(message):
    await asyncio.sleep(2)  # 模拟耗时处理
    print(f"Finish Processing: {message}")
    return f"Finish Processed: {message}"


async def handle_response(websocket, semaphore, message, message_queue):
    async with semaphore:
        print(f"Start Processing: {message}")
        response = await process_message(message)
        await websocket.send(response)
        message_queue.task_done()


async def handler_message(websocket, semaphore, message_queue):
    while True:
        message = await message_queue.get()
        _ = asyncio.create_task(handle_response(websocket, semaphore, message, message_queue))


async def receive_messages(websocket, message_queue):
    try:
        async for message in websocket:
            request_data = GenerateRequest.model_validate_json(message)
            await message_queue.put(request_data)
    except websockets.exceptions.ConnectionClosedError:
        print("Connection closed while receiving messages")
    except Exception as e:
        print(f"Error receiving message: {str(e)}")


async def start_server(websocket):
    message_queue = asyncio.Queue()
    semaphore = asyncio.Semaphore(5)
    # 启动消息处理器和接收消息任务
    message_task = asyncio.create_task(handler_message(websocket, semaphore, message_queue))
    receiver_task = asyncio.create_task(receive_messages(websocket, message_queue))
    try:
        await asyncio.gather(message_task, receiver_task)
    except Exception as e:
        print(f"Error in handle_client: {str(e)}")
    finally:
        message_task.cancel()  # 确保消费者任务被取消
        receiver_task.cancel()   # 确保接收任务也被取消
        await message_queue.join()  # 等待所有队列任务完成


async def main():
    async with websockets.serve(start_server, "localhost", 6543):
        await asyncio.Future()  # 运行服务器，直到显式停止


if __name__ == "__main__":
    asyncio.run(main())
