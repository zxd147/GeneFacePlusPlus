import gc
import json
import os
import socket
import threading
import time
import argparse, glob, tqdm

from contextlib import asynccontextmanager
from typing import List, Optional, Union
import argparse
import asyncio
import json
import logging
import os
import queue
import time
from typing import Optional, Any

import simplejson
import torch
import websockets
from minio import Minio
from websockets import exceptions

from config.uitls import read_json_file
from inference.genefacepp_infer import GeneFace2Infer, get_arg
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from utils.log_utils import logger
from config.uitls import read_json_file


# 请求数据模型
class VideoGenerateRequest(BaseModel):
    project_type: Union[str, int] = Field(..., description="项目类型")
    uid: str = Field(..., description="会话ID")
    character: str = Field(..., description="数字人形象")
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
    code: int = Field(..., description="状态码")
    timestamp: int = Field(..., description="时间戳")


class VideoRetrieveResponse(BaseModel):
    sno: Union[int, str] = Field(default_factory=lambda: int(time.time() * 100), description="序号")
    code: int = Field(0, description="状态码")
    session: Optional[str] = Field(None, description="会话ID")
    timestamp: Optional[int] = Field(None, description="时间戳")
    stream_url: Optional[str] = Field(None, description="视频流URL")
    data: List[VideoData] = Field([], description="视频数据列表")
    msg: Optional[str] = Field('success', description="信息")


class LiveInferenceResponse(BaseModel):
    code: int = Field(..., description="状态码")
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


# 垃圾回收操作执行垃圾回收和 CUDA 缓存清空
def torch_gc():
    gc.collect()
    if torch.cuda.is_available():  # 检查是否可用CUDA
        torch.cuda.empty_cache()  # 清空CUDA缓存
        torch.cuda.ipc_collect()  # 收集CUDA内存碎片


@asynccontextmanager
async def lifespan(graphrag_app: FastAPI):
    # 启动时执行
    try:
        geneface_log.info("启动中...")
        # 初始化系统
        # llm_config, embedder_config = get_config(new_llm_model=None)
        await refresh_models(model_path_dict=all_model_path, character=CURRENT_CHARACTER)
        # 让应用继续运行
        yield
    except Exception as e:
        geneface_log.error(f"启动失败: {e}")
        raise
    finally:
        geneface_log.info("关闭应用...")


async def refresh_models(model_path_dict, character):
    global CURRENT_CHARACTER, model, inp  # 声明这些变量是全局变量
    if not character == CURRENT_CHARACTER or not model:
        geneface_log.info(f"检测到模型发生变化，正在重新初始化为：{character}")
        model_path = model_path_dict[character]
        inp = get_arg(torso_ckpt=model_path)
        # 实例化
        model = GeneFace2Infer(audio2secc_dir=inp["a2m_ckpt"], postnet_dir=inp["postnet_ckpt"],
                               head_model_dir=inp["head_ckpt"], torso_model_dir=inp["torso_ckpt"])
        CURRENT_CHARACTER = character
        geneface_log.info("模型初始化完成")
    return {"status": "success"}


async def get_models(model_path_dict, character):
    global CURRENT_CHARACTER, model_instances, model, inp
    if not character == CURRENT_CHARACTER or not model:
        temp_model_instances = {model_path: model_instances[model_path]
                                for model_path in model_instances if model_path in model_path_dict}
        # 删除 model_instances，释放其占用的内存
        del model_instances
        torch_gc()

        # 遍历 model_path_dict 中的所有模型路径
        for model_name, model_path in model_path_dict.items():
            if model_path not in temp_model_instances:
                inp = get_arg(torso_ckpt=model_path)
                temp_model_instances[model_path] = GeneFace2Infer(audio2secc_dir=inp["a2m_ckpt"],
                                                                  postnet_dir=inp["postnet_ckpt"],
                                                                  head_model_dir=inp["head_ckpt"],
                                                                  torso_model_dir=inp["torso_ckpt"])
            temp_model_instances[model_path] = temp_model_instances[model_path].to('cuda') \
                if model_name == character else temp_model_instances[model_path].to('cpu')
        model_instances = temp_model_instances
        model = model_instances[model_path_dict[character]]
        del temp_model_instances
    return {"status": "success"}


def go_inference(info):
    pass


# 初始化变量
remote_dict = {}
ip_mapping = {}
text_mapping = {}
model_instances = {}  # {model_path: Model}
SNO_START = 123
IDX = 0
inp: Optional[Any] = None
model: Optional[GeneFace2Infer] = None
data = read_json_file('config/config.json')
all_model_path = data['model_path']
base_audio_path = data['base_audio_path']
base_video_path = data['base_video_path']
socket_host = data['socket_ip_host']
socket_port = data['socket_ip_port']
stream_url = data['rtsp_url']
data = read_json_file('config/config.json')
secret_key = os.getenv('GENEFACE-SECRET-KEY', 'sk-geneface')
CURRENT_CHARACTER = "li"
geneface_log = logger
geneface_app = FastAPI()
# CORS 中间件配置
geneface_app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'],
                            allow_headers=['*'], )
# geneface_app.add_middleware(BasicAuthMiddleware, secret_key=secret_key)


@geneface_app.post("/v1/video/generate")
@geneface_app.post("/inference")
async def receive_data(request: VideoGenerateRequest):
    try:
        global CURRENT_CHARACTER
        audio_name = request.audio_name
        character = request.character
        uid = request.uid
        audio_path = os.path.join(base_audio_path, audio_name)
        os.makedirs(os.path.join(base_video_path, str(uid)), exist_ok=True)
        code = 0
        if not os.path.exists(audio_path):
            result = VideoGenerateResponse(id=uid, code=-1, timestamp=int(time.time()))
            return JSONResponse(status_code=200, content=result.model_dump())
        await refresh_models(model_path_dict=all_model_path, character=character)
        # await get_models(model_path_dict=all_model_path, character=character)
        infer_data = {"uid": uid, 'audio_path': audio_path, "project_type": request.project_type}  # 构建ip信息库
        infer_thread = threading.Thread(target=go_inference, args=(infer_data,))
        infer_thread.start()
        result = VideoGenerateResponse(id=uid, code=code, timestamp=int(time.time()))
        return JSONResponse(status_code=200, content=result.model_dump())
    except json.JSONDecodeError as je:
        error_message = VideoGenerateResponse(
            code=-1,
            messages=f"JSONDecodeError, Invalid JSON format: {str(je)} "
        )
        logs = f"Completions response  error: {error_message.model_dump()}"
        geneface_log.error(logs)
        return JSONResponse(status_code=400, content=error_message.model_dump())
    except ValueError as ve:
        error_message = VideoGenerateResponse(
            code=-1,
            messages=f"ValueError, Invalid value encountered: {str(ve)}"
        )
        logs = f"Completions response error: {error_message.model_dump()}"
        geneface_log.error(logs)
        return JSONResponse(status_code=422, content=error_message.model_dump())
    # except Exception as e:
    #     error_message = ChatResponse(
    #         code=-1,
    #         messages=f"Exception, {str(e)}"
    #     )
    #     logs = f"Completions response error: {error_message.model_dump()}"
    #     chat_logger.error(logs)
    #     return JSONResponse(status_code=500, content=error_message.model_dump())
