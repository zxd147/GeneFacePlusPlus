import asyncio
import gc
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Union, Literal
from typing import Optional, Any

import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field, model_validator, ValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

from config.uitls import read_json_file
from inference.genefacepp_infer import GeneFace2Infer, get_arg
from utils.log_utils import logger


# 请求数据模型
class VideoGenerateRequest(BaseModel):
    sno: Union[int, str] = Field(default_factory=lambda: int(time.time() * 100), description="序号")
    uid: str = Field(..., description="会话ID")
    parallel: Optional[bool] = Field(None, description="是否并行")
    character: Literal["huang", "li", "yu"] = Field('huang', description="数字人形象")
    audio_name: str = Field(..., description="音频文件名")
    project_type: Union[str, int] = Field(description="项目类型")


class VideoRetrieveRequest(BaseModel):
    uid: str = Field(..., description="会话ID")
    sno: Union[int, str] = Field(default_factory=lambda: int(time.time() * 100), description="序号")


# class LiveInferenceRequest(BaseModel):
#     sno: Union[int, str] = Field(default_factory=lambda: int(time.time() * 100), description="序号")
#     task_id: Optional[Union[int, str]] = Field(0, description="任务ID")
#     data_type: Optional[str] = Field(description="数据类型")
#     person_id: Optional[str] = Field(..., description="视频源的ID")
#     driven_audio: str = Field(..., description="驱动音频的文件名")


class VideoData(BaseModel):
    video_dir: str = Field(..., description="视频路径")
    video_name: str = Field(..., description="视频文件路径")
    create_time: int = Field(description="视频文件创建时间戳")

    @model_validator(mode="before")
    def set_time(cls, values):
        if values.get('create_time') is None:  # 如果没有提供time，且video_name存在
            # 获取视频文件的创建时间戳
            try:
                video_path = os.path.join(base_video_path, values['video_dir'], values['video_name'])
                values['create_time'] = int(os.path.getctime(video_path))
            except FileNotFoundError:
                pass
        return values


class VideoGenerateResponse(BaseModel):
    sno: Union[int, str] = Field(description="序号")
    type: Optional[Union[int, str]] = Field(description="项目类型")
    code: int = Field(..., description="状态码")
    msg: str = Field(..., description="返回信息")
    data: List[VideoData] = Field([], description="视频数据列表")
    timestamp: int = Field(default_factory=lambda: int(time.time() * 100), description="时间戳")


class VideoRetrieveResponse(BaseModel):
    sno: Union[int, str] = Field(description="序号")
    type: Optional[Union[int, str]] = Field(description="项目类型")
    code: int = Field(0, description="状态码")
    msg: str = Field(..., description="返回信息")
    data: List[VideoData] = Field([], description="视频数据列表")
    timestamp: int = Field(default_factory=lambda: int(time.time() * 100), description="时间戳")


# class LiveInferenceResponse(BaseModel):
#     sno: Union[int, str] = Field( description="序号")
#     code: int = Field(..., description="状态码")
#     msg: str = Field(..., description="返回信息")
#     data: list = Field([], description="视频数据列表")
#     timestamp: int = Field(default_factory=lambda: int(time.time() * 100), description="时间戳")


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
async def lifespan(geneface_app: FastAPI):
    # 启动时执行
    try:
        geneface_log.info("启动中...")
        # 初始化系统
        # llm_config, embedder_config = get_config(new_llm_model=None)
        if parallel is None:
            await refresh_models(CURRENT_CHARACTER)
        else:
            await get_models(CURRENT_CHARACTER, parallel)
        # 让应用继续运行
        yield
    except Exception as e:
        geneface_log.error(f"启动失败: {e}")
        raise
    finally:
        geneface_log.info("关闭应用...")


async def init_app():
    if torch.cuda.is_available():
        log = f'本次加载模型的设备为GPU: {torch.cuda.get_device_name(0)}'
    else:
        log = '本次加载模型的设备为CPU.'
    geneface_log.info(log)
    log = f"Service started!"
    geneface_log.info(log)


async def refresh_models(character):
    global CURRENT_CHARACTER, model, inp  # 声明这些变量是全局变量
    if not character == CURRENT_CHARACTER or not model:
        geneface_log.info(f"检测到模型发生变化，正在重新初始化为：{character}")
        model_path = all_model_path[character]
        model = storage_model_instances.get(model_path) or load_model(model_path=model_path)
        CURRENT_CHARACTER = character
        geneface_log.info("模型初始化完成")
    return {"status": "success"}


async def get_models(character, is_parallel, infer_device=None, storage_device=None, load_device=None):
    global CURRENT_CHARACTER, load_model_instances, storage_model_instances, model
    # 选择推理设备，默认使用 cuda（如果可用），否则使用 cpu
    infer_device = infer_device or ('cuda' if torch.cuda.is_available() else 'cpu')
    # 根据并行推理决定存储设备，默认使用 infer_device，如果未提供 storage_device 则使用 'cpu'
    storage_device = infer_device if is_parallel else (storage_device or 'cpu')
    # 设置加载设备，如果未提供则默认使用 'cpu'
    load_device = load_device or 'cpu'
    # 确保 character 是有效值
    if character not in all_model_path:
        raise ValueError(f"Character '{character}' not found in all_model_path keys.")
    if not model:
        geneface_log.info("检测到未加载模型, 开始加载...")
        # 遍历 model_path_dict 中的所有模型路径, 根据模型名称选择设备和存储实例
        for model_name, model_path in all_model_path.items():
            device, model_instances = (
                (load_device, load_model_instances)
                if model_name != character
                else (infer_device, storage_model_instances)
            )
            geneface_log.info(f"正在加载推理模型{model_name}到{device}...")
            # 加载模型
            load_model(model_path, device, model_instances)
        model = storage_model_instances[all_model_path[character]]
        result = {"status": "success", "message": "models loaded", "character": character, "device": infer_device}
        geneface_log.debug(result)
        CURRENT_CHARACTER = character
    elif not character == CURRENT_CHARACTER:
        geneface_log.info(f"检测到模型发生变化，正在重新初始化推理模型{character}到设备{infer_device}...")
        # 遍历 model_path_dict 中的所有模型路径并放到存储设备
        for model_instance in storage_model_instances.values():
            set_model_to_device(model_instance, storage_device)
        torch_gc()  # 垃圾回收
        new_model_path = all_model_path[character]
        if new_model_path in load_model_instances:
            geneface_log.info(f"正在将推理模型{new_model_path}从{load_device}移动到{infer_device}...")
            storage_model_instances[new_model_path] = load_model_instances.pop(new_model_path)
        geneface_log.info(f"正在加载推理模型{new_model_path}到{infer_device}...")
        model = storage_model_instances.setdefault(new_model_path, load_model(model_path=new_model_path))
        set_model_to_device(model, infer_device)
        model = storage_model_instances.get(new_model_path)
        result = {"status": "success", "message": "change model", "character": character, "device": infer_device}
        geneface_log.debug(result)
        CURRENT_CHARACTER = character
    else:
        result = {"status": "success", "message": "no changes", "character": character, "device": infer_device}
        geneface_log.debug(result)
    return result


def load_model(model_path, device=None, target_dict=None):
    """加载或更新模型，移动到指定设备，并放入目标字典"""
    global inp
    inp = get_arg(torso_ckpt=model_path)
    model_instance = GeneFace2Infer(
        audio2secc_dir=inp["a2m_ckpt"],
        postnet_dir=inp["postnet_ckpt"],
        head_model_dir=inp["head_ckpt"],
        torso_model_dir=inp["torso_ckpt"],
        device=device
    )
    if target_dict is None:
        return model_instance
    else:
        target_dict[model_path] = model_instance


def set_model_to_device(model_instance, device_instance):
    """将模型及其组件移动到指定设备并设置为 eval 模式。"""
    # 先判断模型是否已经在目标设备上，再决定是否执行 .to(device_instance)
    if model_instance.audio2secc_model.device != device_instance:
        model_instance.audio2secc_model.to(device_instance).eval()
    if model_instance.secc2video_model.device != device_instance:
        model_instance.secc2video_model.to(device_instance).eval()
    if model_instance.postnet_model and model_instance.postnet_model.device != device_instance:
        model_instance.postnet_model.to(device_instance).eval()


def go_inference(data):
    logs = f"inference param: {data}"
    geneface_log.info(logs)
    inp['drv_aud'] = data['audio_path']
    inp['out_name'] = data['video_path']
    video_path = model.infer_once(inp)
    return video_path


async def send_heartbeat(websocket):
    while True:
        await asyncio.sleep(30)  # 每30秒发送一次心跳
        await websocket.send_json({"type": "ping"})


# 初始化变量
remote_dict = {}
ip_mapping = {}
text_mapping = {}
storage_model_instances = {}  # {model_path: Model}
load_model_instances = {}  # {model_path: Model}
parallel = False
SNO_START = 123
IDX = 0
inp: Optional[Any] = None
model: Optional[GeneFace2Infer] = None
config_data = read_json_file('config/config.json')
all_model_path = config_data['model_path']
base_audio_path = config_data['base_audio_path']
base_video_path = config_data['base_video_path']
socket_host = config_data['socket_ip_host']
socket_port = config_data['socket_ip_port']
stream_url = config_data['rtsp_url']
secret_key = os.getenv('GENEFACE-SECRET-KEY', 'sk-geneface')
CURRENT_CHARACTER = "li"
geneface_log = logger
geneface_app = FastAPI(lifespan=lifespan)
# CORS 中间件配置
# geneface_app.add_middleware(BasicAuthMiddleware, secret_key=secret_key)
geneface_app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'],
                            allow_headers=['*'], )


@geneface_app.get("/")
async def index():
    service_name = """
        <html> <head> <title>geneface_service</title> </head>
            <body style="display: flex; justify-content: center;"> <h1>geneface_service</h1></body> </html>
        """
    return HTMLResponse(status_code=200, content=service_name)


@geneface_app.get("/http_check")
@geneface_app.get("/health")
async def health():
    """Health check."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    health_data = {"status": "healthy", "timestamp": timestamp}
    # 返回JSON格式的响应
    return JSONResponse(status_code=200, content=health_data)


@geneface_app.post("/v1/video/generate")
@geneface_app.post("/v1/video/inference")
async def generate_video(request: VideoGenerateRequest):
    try:
        global CURRENT_CHARACTER
        logs = f"Inference request param: {request.model_dump()}"
        geneface_log.info(logs)
        sno = request.sno
        uid = request.uid
        character = request.character
        audio_name = request.audio_name
        is_parallel = request.parallel or parallel
        video_dir = os.path.join(base_video_path, str(uid))
        os.makedirs(video_dir, exist_ok=True)
        video_name = f"{str(uuid.uuid4().hex[:8])}.mp4"
        audio_path = os.path.join(base_audio_path, audio_name)
        video_path = os.path.join(video_dir, video_name)
        code = 0
        if not os.path.exists(audio_path):
            msg = f'The audio file does not exist:{audio_path}.'
            raise FileNotFoundError(msg)
        if parallel is None:
            await refresh_models(character=character)
        else:
            await get_models(character, is_parallel)
        infer_data = {"uid": uid, 'audio_path': audio_path, "video_path": video_path}
        # 使用 asyncio.create_task 启动后台任务, Run the task in the background, no need to await
        _ = asyncio.create_task(asyncio.to_thread(go_inference, infer_data))
        # # 放入线程运行
        # infer_thread = threading.Thread(target=go_inference, args=(infer_data,))
        # infer_thread.start()
        msg = "infer starting..."
        video_data = [VideoData(video_dir=video_dir, video_name=video_name)]
        result = VideoGenerateResponse(sno=sno, type=1, code=code, msg=msg, data=video_data)
        logs = f"Inference response results: {result.model_dump()}"
        geneface_log.info(logs)
        return JSONResponse(status_code=200, content=result.model_dump())
    except json.JSONDecodeError as je:
        error_message = VideoGenerateResponse(
            code=-1,
            messages=f"JSONDecodeError, Invalid JSON format: {str(je)} "
        )
        logs = f"Inference response  error: {error_message.model_dump()}"
        geneface_log.error(logs)
        return JSONResponse(status_code=400, content=error_message.model_dump())
    except FileNotFoundError as fe:
        error_message = VideoGenerateResponse(
            code=1,
            messages=f'FileNotFoundError: {fe}'
        )
        logs = f"Inference response error: {error_message.model_dump()}"
        geneface_log.error(logs)
        return JSONResponse(status_code=404, content=error_message.model_dump())
    except ValueError as ve:
        error_message = VideoGenerateResponse(
            code=-1,
            messages=f"ValueError, Invalid value encountered: {str(ve)}"
        )
        logs = f"Inference response error: {error_message.model_dump()}"
        geneface_log.error(logs)
        return JSONResponse(status_code=422, content=error_message.model_dump())
    # except Exception as e:
    #     error_message = ChatResponse(
    #         code=-1,
    #         messages=f"Exception, {str(e)}"
    #     )
    #     logs = f"Inference response error: {error_message.model_dump()}"
    #     chat_logger.error(logs)
    #     return JSONResponse(status_code=500, content=error_message.model_dump())


@geneface_app.post("/v1/video/retrieve")
@geneface_app.post("/v1/video/get")
async def retrieve_video(request: VideoRetrieveRequest):
    try:
        logs = f"Retrieve request param: {request.model_dump()}"
        geneface_log.info(logs)
        sno = request.sno
        uid = request.uid
        video_dir = os.path.join(base_video_path, str(uid))
        video_files = [file for file in os.listdir(video_dir) if
                       file.endswith('.mp4') and not file.endswith('.tmp.mp4')]
        sorted_files = sorted(video_files, key=lambda x: os.path.getmtime(os.path.join(video_dir, x)), reverse=True)[:5]
        (code, msg) = (0, f"Successfully retrieved {len(sorted_files)} video files") if sorted_files else (
            1, "No videos found")
        video_data = [VideoData(video_name=video_name, video_dir=video_dir) for video_name in sorted_files]
        result = VideoRetrieveResponse(
            sno=sno,
            type=2,
            code=code,
            msg=msg,
            data=video_data)
        logs = f"Retrieve response results: {result.model_dump()}"
        geneface_log.info(logs)
        return JSONResponse(status_code=200, content=result.model_dump())
    except json.JSONDecodeError as je:
        error_message = VideoRetrieveResponse(
            code=-1,
            messages=f"JSONDecodeError, Invalid JSON format: {str(je)} "
        )
        logs = f"Retrieve response  error: {error_message.model_dump()}"
        geneface_log.error(logs)
        return JSONResponse(status_code=400, content=error_message.model_dump())
    except FileNotFoundError as fe:
        error_message = VideoRetrieveResponse(
            code=1,
            messages=f'FileNotFoundError: The video directory does not exist: {fe}'
        )
        logs = f"Retrieve response error: {error_message.model_dump()}"
        geneface_log.error(logs)
        return JSONResponse(status_code=404, content=error_message.model_dump())
    except ValueError as ve:
        error_message = VideoRetrieveResponse(
            code=-1,
            messages=f"ValueError, Invalid value encountered: {str(ve)}"
        )
        logs = f"Retrieve response error: {error_message.model_dump()}"
        geneface_log.error(logs)
        return JSONResponse(status_code=422, content=error_message.model_dump())
    # except Exception as e:
    #     error_message = VideoRetrieveResponse(
    #         code=-1,
    #         messages=f"Exception, {str(e)}"
    #     )
    #     logs = f"Retrieve response error: {error_message.model_dump()}"
    #     chat_logger.error(logs)
    #     return JSONResponse(status_code=500, content=error_message.model_dump())


# 定义 WebSocket 路由
@geneface_app.websocket("/ws/video/generate")
@geneface_app.websocket("/ws/video/inference")
async def websocket_endpoint(websocket: WebSocket):
    client_host = websocket.client.host
    client_port = websocket.client.port
    # 记录客户端连接信息
    connect_msg = f"WebSocket connected from {client_host}:{client_port}"
    geneface_log.info(connect_msg)
    # 接受连接
    await websocket.accept()
    try:
        global CURRENT_CHARACTER
        while True:
            # 接收客户端发送的数据
            data = await websocket.receive_json()  # 自动解析接收到的 JSON 数据并将其转换为字典
            request = VideoGenerateRequest.model_validate_json(data)
            logs = f"Inference request param: {request.model_dump()}"
            geneface_log.info(logs)

            sno = request.sno
            uid = request.uid
            character = request.character
            audio_name = request.audio_name
            is_parallel = request.parallel or parallel
            video_dir = os.path.join(base_video_path, str(uid))
            os.makedirs(video_dir, exist_ok=True)
            video_name = f"{str(uuid.uuid4().hex[:8])}.mp4"
            audio_path = os.path.join(base_audio_path, audio_name)
            video_path = os.path.join(video_dir, video_name)

            # 校验音频文件是否存在
            if not os.path.exists(audio_path):
                msg = f'The audio file does not exist: {audio_path}.'
                error_message = {
                    "code": 1,
                    "messages": f'FileNotFoundError: {msg}',
                }
                logs = f"Inference response error: {error_message}"
                geneface_log.error(logs)
                await websocket.send_json(error_message)
                continue

            # 刷新或加载模型
            if parallel is None:
                await refresh_models(character=character)
            else:
                await get_models(character, is_parallel)
            # 返回推理开始的消息
            code = 0
            msg = "Inference starting..."
            video_data = [VideoData(video_dir=video_dir, video_name=video_name)]
            result = VideoGenerateResponse(sno=sno, type=1, code=code, msg=msg, data=video_data)
            logs = f"Inference response results: {result.model_dump()}"
            geneface_log.info(logs)
            await websocket.send_json(result.model_dump())

            infer_data = {"uid": uid, "audio_path": audio_path, "video_path": video_path}
            # # 启动推理线程
            # infer_thread = threading.Thread(target=go_inference, args=(infer_data,))
            # infer_thread.start()
            # 使用 asyncio.to_thread 调用推理函数

            await asyncio.to_thread(go_inference, infer_data)
            msg = "Inference success"
            code = 0
            video_data = [VideoData(video_dir=video_dir, video_name=video_name)]
            result = VideoRetrieveResponse(
                sno=sno,
                type=2,
                code=code,
                msg=msg,
                data=video_data)
            logs = f"Retrieve response results: {result.model_dump()}"
            geneface_log.info(logs)
            await websocket.send_json(result.model_dump())
    except json.JSONDecodeError as je:
        error_message = {
            "code": -1,
            "messages": f"JSONDecodeError, Invalid JSON format: {str(je)}",
        }
        logs = f"Inference response error: {error_message}"
        geneface_log.error(logs)
        await websocket.send_json(error_message)
    except ValidationError as ve:
        # 捕获 Pydantic 的验证错误
        error_message = f"Validation error: {ve.errors()}"
        geneface_log.error(error_message)
        # 发送错误信息给客户端
        await websocket.send_text(f"Error: {error_message}")
    except FileNotFoundError as fe:
        error_message = {
            "code": 1,
            "messages": f'FileNotFoundError: {fe}',
        }
        logs = f"Inference response error: {error_message}"
        geneface_log.error(logs)
        await websocket.send_json(error_message)
    except ValueError as ve:
        error_message = {
            "code": -1,
            "messages": f"ValueError, Invalid value encountered: {str(ve)}",
        }
        logs = f"Inference response error: {error_message}"
        geneface_log.error(logs)
        await websocket.send_json(error_message)
    except WebSocketDisconnect:
        msg = f"WebSocket disconnected from {client_host}:{client_port}"
        geneface_log.info(msg)
        # 如果需要发送消息或执行额外清理操作，可以在这里添加
        # await websocket.close()  # 确保 WebSocket 已正确关闭


if __name__ == "__main__":
    asyncio.run(init_app())
    uvicorn.run(geneface_app, host="0.0.0.0", port=8041)
