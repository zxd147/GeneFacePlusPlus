import argparse
import asyncio
import json
import logging
import os
import queue
import socket
import time

import websockets
from minio import Minio
from websockets import exceptions

from config.uitls import read_json_file
from inference.genefacepp_infer import GeneFace2Infer

'''
2024.5.22 新增线上推流服务
2024.6.6 需要新增队列维护内容
'''

# 创建日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# 创建文件处理器
log_dir = 'logs'  # 日志文件保存目录
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'out.log')
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
# 创建格式化器
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
# 添加文件处理器到日志记录器
logger.addHandler(file_handler)

isBusy = False
gWsk_remote = None
data_json = read_json_file('config/config.json')
socket_ip = data_json['socket_ip_host']
socket_port = data_json['socket_ip_port']
base_audio_path = data_json['base_audio_path']
base_video_path = data_json['base_video_path']
access_key = data_json['access_key']
secret_key = data_json['secret_key']
upload_server = data_json['api_server_name']
message_queue = queue.Queue()  # add
inference_queue = queue.Queue()  # add


async def websocket_handler(websocket, path):
    global gWsk_remote
    # 当有新的 WebSocket 连接建立时
    print(f"New connection from {websocket.remote_address} path {path}")

    if gWsk_remote is None:
        gWsk_remote = websocket
    else:
        print(f"重复连接：New connection from {websocket.remote_address} path {path}")
        # await websocket.send("-1")
        # await websocket.close()
        return
    try:
        # 循环监听 WebSocket 客户端发送的消息
        async for message in websocket:
            print(f"Received message: {message}")
            '''
            try:
                cur_json = json.loads(message)
                sId = cur_json["id_path"]
                print(f"sId={sId}")
            except Exception as e:
                print(f"读取json信息【错误】{e}")
                # return

            # 原样返回收到的消息
            await websocket.send(message)
            '''

    except websockets.exceptions.ConnectionClosedError:
        print(f"Connection to {websocket.remote_address} closed")
        gWsk_remote = None
        print(f"客户端关闭")


# 处理客户端连接
async def handle_client(reader, writer):
    data = await reader.read(2048)  # 1024
    message = data.decode("utf-8")
    print(f'Received: {message}')
    logger.info(f'{time.time()} 当前收到推理信息 {message}')
    # 将接收到的消息放入队列
    message_queue.put(message)
    writer.close()
    await writer.wait_closed()


# socket服务端
async def start_server():
    server = await asyncio.start_server(handle_client, socket_ip, int(socket_port))
    async with server:
        await server.serve_forever()


# 从队列中取出消息,进行推理
async def process_messages():
    while True:
        try:
            if not message_queue.empty():
                message = message_queue.get()
                # if message is not None:
                #     await inference(message)
                # else:
                #     print('从队列中取出的消息为 None,跳过处理')
                #     pass

                if not message:
                    await asyncio.sleep(1)  # 暂停 1 秒钟
                else:
                    await inference(message)

            else:
                # print('当前队列为空，稍等')
                pass
        except Exception as e:
            print("Error 出现的错误", e)
        await asyncio.sleep(0.5)  # 暂停 1 秒钟


async def main():
    # websocket服务端开启服务
    websocket_host = "0.0.0.0"
    websocket_port = 5465
    start_websocket = websockets.serve(websocket_handler, websocket_host, websocket_port)
    print(f"WebSocket server started, Listening on ws://{websocket_host}:{websocket_port}")
    await asyncio.gather(
        start_server(),
        process_messages(),
        start_websocket
    )


'''---------------模型推理内容---------------'''


def parameter():
    global data_json
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2m_ckpt",
                        default='checkpoints/audio2motion_vae')
    parser.add_argument("--head_ckpt", default='')
    parser.add_argument("--postnet_ckpt", default='')
    parser.add_argument("--torso_ckpt", default='checkpoints/motion2video_nerf/may_torso')
    parser.add_argument("--drv_aud", default='data/raw/val_wavs/MacronSpeech.wav')
    parser.add_argument("--drv_pose", default='nearest',
                        help="目前仅支持源视频的pose,包括从头开始和指定区间两种,暂时不支持in-the-wild的pose")
    parser.add_argument("--blink_mode", default='none')  # none | period
    parser.add_argument("--lle_percent", default=0.2)  # nearest | random
    parser.add_argument("--temperature", default=0.2)  # nearest | random
    parser.add_argument("--mouth_amp", default=0.3)  # nearest | random
    parser.add_argument("--raymarching_end_threshold", default=0.01,
                        help="increase it to improve fps")  # nearest | random
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--fast", action='store_true')
    parser.add_argument("--out_name", default='tmp.mp4')
    parser.add_argument("--low_memory_usage", action='store_true',
                        help='write img to video upon generated, leads to slower fps, but use less memory')

    args = parser.parse_args()
    # 静态
    args.a2m_ckpt = data_json['a2m_ckpt']
    args.torso_ckpt = data_json['torso_ckpt']
    args.drv_aud = data_json['drv_aud']
    inp = {
        'a2m_ckpt': args.a2m_ckpt,
        'postnet_ckpt': args.postnet_ckpt,
        'head_ckpt': args.head_ckpt,
        'torso_ckpt': args.torso_ckpt,
        'drv_audio_name': args.drv_aud,
        'drv_pose': args.drv_pose,
        'blink_mode': args.blink_mode,
        'temperature': float(args.temperature),
        'mouth_amp': args.mouth_amp,
        'lle_percent': float(args.lle_percent),
        'debug': args.debug,
        'out_name': args.out_name,
        'raymarching_end_threshold': args.raymarching_end_threshold,
        'low_memory_usage': args.low_memory_usage,
    }
    print(f"使用的推理模型为{inp['torso_ckpt']}")
    #    exit(0)
    if args.fast:
        inp['raymarching_end_threshold'] = 0.05
    # GeneFace2Infer.example_run(inp) # 可以不实例化
    return args, inp


args, inp = parameter()
obj = GeneFace2Infer(audio2secc_dir=args.a2m_ckpt, postnet_dir=args.postnet_ckpt, head_model_dir=args.head_ckpt,
                     torso_model_dir=args.torso_ckpt)
IDX = 0


# def inference(line):
async def inference(line):
    global IDX, isBusy
    logger.info(f'{time.time()} 当前推理信息 {line} 序号 {IDX}')
    isBusy = True
    cur_info = line.replace("'", "\"")
    try:
        cur_json = json.loads(cur_info)
    except json.JSONDecodeError as e:
        print(f"【时间】{time.ctime()}  读取json信息【错误】{e}")
        return

    inp['drv_aud'] = os.path.join(base_audio_path, cur_json['audio'])
    video_output = str(int(time.time())) + ".mp4"
    id = cur_json["session_id"]

    project_type = cur_json['project_type']

    id_path = os.path.join(base_video_path, str(id))
    os.makedirs(id_path, exist_ok=True)
    inp['out_name'] = os.path.join(id_path, video_output)

    try:
        # obj.infer_once_v2(inp,project_type) # 推理
        obj.infer_once(inp)  # 推理

        if project_type == 1:
            # 进行文件的上传
            video_save_path = inp['out_name']
            upload(video_save_path)
    except Exception as e:
        print(f"【时间】{time.ctime()} 推理错误 当前内容 {inp} 【错误】{e}")

    print('推理结束')
    IDX += 1
    isBusy = False
    global gWsk_remote
    if gWsk_remote is None:
        print(f"客户端未连接！")
    else:
        await gWsk_remote.send(f"{id}/{video_output}")


'''------------------------------------'''

'''================文件上传内容==============='''

print("upload_server", upload_server)
client = Minio(upload_server,
               access_key=access_key,
               secret_key=secret_key,
               region="cn-south-1",
               secure=True
               )


def upload(source_file):
    bucket_name = "customer"
    content_type = "video/mp4"

    destination_file = source_file.replace(base_video_path, "video")
    found = client.bucket_exists(bucket_name)
    if not found:
        client.make_bucket(bucket_name)
        print("Created bucket", bucket_name)
    try:
        client.fput_object(bucket_name, destination_file, source_file, content_type)
        print("文件上传成功", source_file, )
    except Exception as e:
        print("文件上传失败")


'''=========================================='''

if __name__ == "__main__":
    # start_server()
    asyncio.run(main())
