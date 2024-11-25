import json
import os
import socket
import time
from typing import List, Optional, Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

from config.uitls import read_json_file


# 请求数据模型
class VideoGenerateRequest(BaseModel):
    project_type: Union[str, int] = Field(..., description="项目类型")
    uid: str = Field(..., description="会话ID")
    audio_name: str = Field(..., description="音频文件名")


class VideoRetrieveRequest(BaseModel):
    uid: str = Field(..., description="会话ID")
    sno: Optional[int] = Field(None, description="序号")


class LiveInferenceRequest(BaseModel):
    task_id: Optional[str] = Field(None, description="任务ID")
    data_type: Optional[str] = Field(None, description="数据类型")
    person_id: Optional[str] = Field(None, description="视频源的ID")
    driven_audio: str = Field(..., description="驱动音频的文件名")


class VideoData(BaseModel):
    uid: str = Field(..., description="会话ID")
    time: int = Field(..., description="视频文件创建时间戳")
    video_name: str = Field(..., description="视频文件路径")


class VideoGenerateResponse(BaseModel):
    id: str = Field(..., description="会话ID")
    status: int = Field(..., description="状态码")
    timestamp: int = Field(..., description="时间戳")


class VideoRetrieveResponse(BaseModel):
    sno: Union[int, str] = Field(default_factory=lambda: int(time.time() * 100), description="序号")
    status: int = Field(0, description="状态码")
    session: Optional[str] = Field(None, description="会话ID")
    timestamp: Optional[int] = Field(None, description="时间戳")
    stream_url: Optional[str] = Field(None, description="视频流URL")
    data: List[VideoData] = Field([], description="视频数据列表")
    msg: Optional[str] = Field('success', description="信息")


class LiveInferenceResponse(BaseModel):
    status: int = Field(..., description="状态码")
    msg: str = Field(..., description="返回信息")
    timestamp: int = Field(..., description="时间戳")


class BasicAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, secret_key: str):
        super().__init__(app)
        self.required_credentials = secret_key

    async def dispatch(self, request: StarletteRequest, call_next):
        authorization: str = request.headers.get('Authorization')
        if authorization and authorization.startswith('Bearer '):
            provided_credentials = authorization.split(' ')[1]
            # 比较提供的令牌和所需的令牌
            if provided_credentials == self.required_credentials:
                return await call_next(request)
        # 返回一个带有自定义消息的JSON响应
        return JSONResponse(
            status_code=400,
            content={"detail": "Unauthorized: Invalid or missing credentials"},
            headers={'WWW-Authenticate': 'Bearer realm="Secure Area"'}
        )


geneface_app = FastAPI()
secret_key = os.getenv('GENEFACE-SECRET-KEY', 'sk-geneface')
# CORS 中间件配置
geneface_app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'], )
# geneface_app.add_middleware(BasicAuthMiddleware, secret_key=secret_key)
# 初始化变量
remote_dict = {}
ip_mapping = {}
text_mapping = {}
SNO_START = 123
data = read_json_file('config/config.json')
base_audio_path = data['base_audio_path']
base_video_path = data['base_video_path']
socket_host = data['socket_ip_host']
socket_port = data['socket_ip_port']
stream_url = data['rtsp_url']


def send_data_to_modeler(inference_info):
    # 创建一个新的套接字对象, AF_INET 表示使用 IPv4 地址, SOCK_STREAM 表示这是一个流式套接字（TCP）
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(2)  # 设置2s超时时间
    cur_id = inference_info['session_id']
    os.makedirs(os.path.join(base_video_path, str(cur_id)), exist_ok=True)
    print(f"{time.ctime()} 当前发送的推理信息 {inference_info}")  # 'time.ctime(): Wed Oct  9 17:04:28 2024'

    try:
        client_socket.connect((socket_host, int(socket_port)))
        # 将字典转换为 JSON 字符串并发送
        inference_info_json = json.dumps(inference_info)  # 转换为 JSON 字符串
        client_socket.send(inference_info_json.encode("utf-8"))  # 编码为字节
        # 接收服务器的响应
        return_data = client_socket.recv(1024).decode('utf-8')
        if return_data:
            print("接受socket返回", return_data)
    except socket.timeout:
        print("连接超时")
    finally:
        client_socket.close()


@geneface_app.post("/v1/video/generate")
@geneface_app.post("/inference")
async def receive_data(request: VideoGenerateRequest):
    audio_name = request.audio_name
    project_type = request.project_type
    session_id = request.uid

    global remote_dict
    if session_id == "dentist":
        remote_dict.setdefault(session_id, request.client.host)
        print(f"remote = {remote_dict[session_id]}"
              if remote_dict[session_id] == request.client.host else "remote 已保存！")
    wav_path = os.path.join(base_audio_path, audio_name)
    if os.path.exists(wav_path):
        status = 0
        ip_mapping.setdefault(session_id, len(ip_mapping) + 1)
        info = {"session_id": session_id, 'audio': audio_name, "project_type": project_type}  # 构建ip信息库 ，并将信息传入到队列
        send_data_to_modeler(info)
    else:
        status = -1
    result = VideoGenerateResponse(id=session_id, status=status, timestamp=int(time.time()))
    return JSONResponse(status_code=200, content=result.model_dump())


@geneface_app.post("/v1/video/retrieve")
@geneface_app.post("/get_video")
async def return_files_list(request: VideoRetrieveRequest):
    global SNO_START
    data_json = request.model_dump()
    if 'uid' not in data_json:
        result = VideoRetrieveResponse(status=1, timestamp=int(time.time()), msg="no such uid")
        return result.model_dump()

    ask_id = request.uid
    sno = request.sno or SNO_START + 1
    video_path = os.path.join(base_video_path, str(ask_id))
    if not os.path.exists(video_path):
        result = VideoRetrieveResponse(status=1, timestamp=int(time.time()), msg=f"no such video_path {video_path}")
        return result.model_dump()
    files = os.listdir(video_path)
    sound_files = [os.path.join(video_path, file) for file in files if file.endswith('.mp4')]
    sorted_files = sorted(sound_files, key=lambda x: os.path.getmtime(x), reverse=True)
    sorted_files_top5 = sorted_files[:5]
    result = VideoRetrieveResponse(
        sno=sno, timestamp=int(time.time()), session=ask_id, stream_url=stream_url,
        data=[
            VideoData(
                video_name=filename.replace(base_video_path, "video"),
                time=int(os.path.getctime(filename)),
                uid=ask_id
            )
            for filename in sorted_files_top5 if filename.split(".")[-2] != 'tmp'
        ]
    )
    return result


@geneface_app.post('v1//live/inference')
async def live_info(request: LiveInferenceRequest):
    inference_data = request.model_dump()
    source_video = request.person_id
    data_type = request.data_type
    task_id = request.task_id
    driven_audio = request.driven_audio
    wav_path = os.path.join(base_audio_path, driven_audio)
    msg = f'got audio file: {wav_path}' if os.path.exists(wav_path) else f'audio file not found: {wav_path}'
    # 推理的进行没写
    result = LiveInferenceResponse(status=0, msg=msg, timestamp=int(time.time()))
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(geneface_app, host="0.0.0.0", port=8041)
