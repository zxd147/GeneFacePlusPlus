import asyncio
import gc
import json
import os
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Union, Literal, Dict
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field, model_validator, ValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

from crop_paste.paste_back import paste_back_by_ffmpeg
from inference.genefacepp_infer import GeneFace2Infer, get_arg
from utils.log_utils import logger
from utils.uitls import read_json_file


# 请求数据模型
class VideoGenerateRequest(BaseModel):
    sno: Union[int, str] = Field(default_factory=lambda: int(time.time() * 100), description="序号")
    uid: str = Field(..., description="会话ID")
    parallel: Optional[bool] = Field(None, description="是否并行")
    character: Literal["huang", "li", "yu", "gu"] = Field('huang', description="数字人形象")
    paste: str = Field('default', description="贴回方式")
    blocking: bool = Field(False, description="是否阻塞运行推理代码")
    audio_name: str = Field(..., description="音频文件名")


class VideoRetrieveRequest(BaseModel):
    uid: str = Field(..., description="会话ID")
    sno: Union[int, str] = Field(default_factory=lambda: int(time.time() * 100), description="序号")
    video_num: Optional[int] = Field(5, description="返回视频个数")


# class LiveInferenceRequest(BaseModel):
#     sno: Union[int, str] = Field(default_factory=lambda: int(time.time() * 100), description="序号")
#     task_id: Optional[Union[int, str]] = Field(0, description="任务ID")
#     data_type: Optional[str] = Field(description="数据类型")
#     person_id: Optional[str] = Field(..., description="视频源的ID")
#     driven_audio: str = Field(..., description="驱动音频的文件名")


class VideoData(BaseModel):
    video_dir: str = Field(..., description="视频路径")
    video_name: str = Field(..., description="视频文件路径")
    create_time: Optional[int] = Field(None, description="视频文件创建时间戳")

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
    sno: Optional[Union[int, str]] = Field(None, description="序号")
    phase: Optional[Union[int, str]] = Field(None, description="阶段")
    code: int = Field(..., description="状态码")
    msg: str = Field(..., description="返回信息")
    data: List[VideoData] = Field([], description="视频数据列表")
    timestamp: int = Field(default_factory=lambda: int(time.time() * 100), description="时间戳")


class VideoRetrieveResponse(BaseModel):
    sno: Optional[Union[int, str]] = Field(None, description="序号")
    phase: Optional[Union[int, str]] = Field(None, description="阶段")
    code: int = Field(..., description="状态码")
    msg: str = Field(..., description="返回信息")
    data: List[VideoData] = Field([], description="视频数据列表")
    timestamp: int = Field(default_factory=lambda: int(time.time() * 100), description="时间戳")


# class LiveInferenceResponse(BaseModel):
#     sno: Optional[Union[int, str]] = Field(None, description="序号")
#     code: int = Field(..., description="状态码")
#     msg: str = Field(..., description="返回信息")
#     data: list = Field([], description="视频数据列表")
#     timestamp: int = Field(default_factory=lambda: int(time.time() * 100), description="时间戳")


class BasicAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, sk: str):
        super().__init__(app)
        self.required_credentials = sk

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
        load_para = get_arg()
        load_para['character'] = CURRENT_CHARACTER
        load_para['torso_ckpt'] = inference_info[CURRENT_CHARACTER]['torso_ckpt']
        load_para['head_ckpt'] = inference_info[CURRENT_CHARACTER]['head_ckpt']
        if parallel is None:
            await refresh_models(load_para, parallel)
        else:
            await get_models(load_para, parallel)
        # 让应用继续运行
        yield
    except Exception as e:
        geneface_log.error(f"启动失败: {e}")
        raise
    finally:
        geneface_log.info("关闭应用...")


async def init_app():
    if torch.cuda.is_available():
        log = f'本次加载推理模型的设备为GPU: {torch.cuda.get_device_name(0)}.\n'
    else:
        log = '本次加载推理模型的设备为CPU.\n'
    log += f"Service start...!"
    geneface_log.info(log)


async def refresh_models(load_para, is_parallel=None, infer_device=None):
    global CURRENT_CHARACTER, model  # 声明这些变量是全局变量
    character = load_para['character']
    if is_parallel is None:
        infer_device = infer_device or ('cuda' if torch.cuda.is_available() else 'cpu')
        if character in inference_info and (not character == CURRENT_CHARACTER or not model):
            if not model:
                inference_info[character]['infer_queue'] = queue.Queue()
                inference_info[character]['queue_lock'] = threading.Lock()
                inference_info[character]['target_device'] = infer_device
            geneface_log.info(f"检测到模型发生变化，正在重新初始化为：{character}")
            model = await load_model(load_para)
            torch_gc()
            CURRENT_CHARACTER = character
            geneface_log.info("模型初始化完成")
        return {"status": "success"}


async def get_models(infer_para, is_parallel, infer_device=None, storage_device=None, init_device=None):
    new_character = infer_para['character']
    new_model_path = infer_para['torso_ckpt'] or infer_para['head_ckpt']
    if is_parallel is not None:
        global CURRENT_CHARACTER, init_model_instances, storage_model_instances
        # 选择推理设备，默认使用 cuda（如果可用），否则使用 cpu
        infer_device = infer_device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # 根据并行推理决定存储设备，默认使用 infer_device，如果未提供 storage_device 则使用 'cpu'
        storage_device = infer_device if is_parallel else (storage_device or 'cpu')
        # 设置加载设备，如果未提供则默认使用 'cpu'
        init_device = init_device or 'cpu'
        # 确保 character 是有效值
        if new_character not in inference_info:
            raise ValueError(f"New Character '{new_character}' not found in inference_info keys.")
        if not storage_model_instances:
            geneface_log.info("检测到未存储模型, 开始加载...")
            # 遍历 model_path_dict 中的所有模型路径, 根据模型名称选择设备和存储实例
            for character_name, character_info in inference_info.items():
                load_para = get_arg()
                load_para['character'] = character_name
                load_para['torso_ckpt'] = character_info['torso_ckpt']
                load_para['head_ckpt'] = character_info['head_ckpt']
                model_path = load_para['torso_ckpt'] or load_para['head_ckpt']
                storage_model_instances[model_path] = None
                load_device, target_dict = (
                    (init_device, init_model_instances)
                    if character_name != new_character
                    else (infer_device, storage_model_instances)
                )
                character_info['infer_queue'] = queue.Queue()
                character_info['queue_lock'] = threading.Lock()
                character_info['target_device'] = load_device
                # 加载模型
                await load_model(load_para, load_device, target_dict)

            # 计算模型总数
            # total_models = len(storage_model_instances) + len(init_model_instances)
            load_models_count = sum(1 for v in init_model_instances.values() if v is not None)
            storage_models_count = sum(1 for v in storage_model_instances.values() if v is not None)
            total_models_count = load_models_count + storage_models_count
            # 打印成功加载的模型个数
            geneface_log.info(f"成功加载模型个数: {total_models_count}个, 当前使用的模型为{new_character}.")
            inference_info[new_character]['model'] = storage_model_instances[new_model_path]
            torch_gc()
            CURRENT_CHARACTER = new_character
            result = {"status": "success", "message": "models loaded", "character": new_character, "device": infer_device}
            geneface_log.debug(result)
        elif not new_character == CURRENT_CHARACTER:
            geneface_log.info(f"检测到模型发生变化，正在重新初始化推理模型{new_character}到设备{infer_device}...")
            if new_model_path in init_model_instances:
                geneface_log.info(
                    f"正在将推理模型{Path(new_model_path).name}, 从 storage_model_instances 移动到 init_model_instances ...")
                storage_model_instances[new_model_path] = init_model_instances.pop(new_model_path)
            # 遍历 model_path_dict 中的所有模型路径并放到存储设备
            move_tasks = []
            # for model_path, model_instance in storage_model_instances.items():
            for character_name, character_info in inference_info.items():
                model_path = character_info['torso_ckpt'] or character_info['head_ckpt']
                model_instance = storage_model_instances[model_path]
                if character_name == new_character or is_parallel or model_instance is None:
                    continue
                inference_info[new_character]["target_device"] = storage_device
                geneface_log.info(f"等待移动推理模型{Path(model_path).name}到{storage_device}...")
                task = set_model_to_device(character_name, model_path, model_instance, storage_device, thread=True)
                move_tasks.append(asyncio.wait_for(task, timeout=300))
                # _ = asyncio.create_task(set_model_to_device(model_path, model_instance, storage_device, thread=True))
                # _ = asyncio.to_thread(set_model_to_device, model_path, model_instance, storage_device, thread=True)
                # loop = asyncio.get_event_loop()
                # asyncio.run_coroutine_threadsafe(set_model_to_device(model_path, model_instance, storage_device, wait=True), loop)
            _ = asyncio.gather(*move_tasks)
            geneface_log.info(f"正在移动推理模型{Path(new_model_path).name}到{infer_device}...")
            inference_info[new_character]["target_device"] = infer_device
            inference_info[new_character]['model'] = storage_model_instances[new_model_path]
            await set_model_to_device(new_character, new_model_path, storage_model_instances[new_model_path], infer_device)
            torch_gc()  # 垃圾回收
            CURRENT_CHARACTER = new_character
            result = {"status": "success", "message": "change model", "character": new_character, "device": infer_device}
            geneface_log.debug(result)
        else:
            result = {"status": "success", "message": "no changes", "character": new_character, "device": infer_device}
            geneface_log.debug(result)
        return result


async def load_model(load_para, load_device=None, target_dict=None):
    """加载或更新模型，移动到指定设备，并放入目标字典"""
    model_instance = GeneFace2Infer(torso_model_dir=load_para['torso_ckpt'], head_model_dir=load_para['head_ckpt'],
                                    device=load_device)
    model_path = load_para['torso_ckpt'] or load_para['head_ckpt']
    geneface_log.info(f"正在加载推理模型{Path(model_path).name}到{load_device}...")

    # 提前返回：如果没有传入 target_dict，则直接返回实例
    if target_dict is None:
        return model_instance
    target_dict[model_path] = model_instance
    return target_dict


async def set_model_to_device(character_name, model_path, model_instance, device_instance, thread=False):
    """将模型及其组件移动到指定设备并设置为 eval 模式。"""

    def to_device():
        # 遍历每个模型组件并移动到目标设备，且设置为eval模式
        for sub_model in sub_models:
            if sub_model:  # 确保模型存在
                geneface_log.debug(
                    f"-------开始移动-------{model_path}, {infer_queue.qsize()}, {device_instance}, {thread}-------开始移动-------")
                sub_model.to(target_device).eval()
                geneface_log.debug(
                    f"=======移动完成======={model_path}, {infer_queue.qsize()}, {device_instance}, {thread}=======移动完成=======")
                torch_gc()

    infer_queue = inference_info[character_name].get("infer_queue")
    target_device = inference_info[character_name]["target_device"]
    # 定义所有模型组件的列表
    sub_models = [model_instance.audio2secc_model, model_instance.secc2video_model, model_instance.postnet_model]
    if model_instance.audio2secc_model.device != target_device:
        geneface_log.debug(
            f"=======准备移动======={model_path}, {infer_queue.qsize()}, {device_instance}, {thread}-------准备移动-------")
        if thread:
            if not infer_queue.empty():  # 如果队列不为空，说明有任务正在进行
                msg = f'Model "{Path(model_path).name}" is busy. Waiting for the current task to finish.'
                geneface_log.info(msg)
                # infer_queue.join()  # 等待队列完成当前任务
                try:
                    await asyncio.wait_for(asyncio.to_thread(infer_queue.join), timeout=200)
                    # await asyncio.to_thread(infer_queue.join)  # 放到异步线程等待队列完成当前任务
                except Exception as e:
                    return
            # _ = asyncio.wait_for(asyncio.to_thread(to_device), timeout=30)  # 同步会导致阻塞卡死
            _ = asyncio.to_thread(to_device)
        else:
            to_device()
    else:
        geneface_log.info(
            f"模型{Path(model_path).name}已在目标设备, 无需移动")


async def run_inference_async(infer_para):
    paste = infer_para['paste']
    character = infer_para['character']
    blocking = infer_para['blocking']
    final_video_path = infer_para['final_video_path']
    infer_video_path = infer_para['out_name']
    ori_video_path = inference_info[character]['ori_video']
    crop_coordinates_list = inference_info[character]['crop_coordinates']
    logs = f"inference parameters: {infer_para}"
    geneface_log.info(logs)
    loop = asyncio.get_event_loop()
    task = loop.run_in_executor(thread_executor, infer_in_executor, infer_para)
    _ = await task if blocking else None
    # final_video_path = final_video_path if paste == 'default' else paste_back_by_ffmpeg(infer_video_path, ori_video_path, final_video_path, crop_coordinates_list)
    final_video_path = final_video_path if paste == 'default' else await asyncio.to_thread(paste_back_by_ffmpeg, infer_video_path, ori_video_path, final_video_path, crop_coordinates_list)
    return final_video_path


def infer_in_executor(infer_para):
    character = infer_para['character']
    is_parallel = infer_para['parallel']
    # 队列逻辑确保只有一个任务进行
    infer_queue = inference_info[character]["infer_queue"]
    infer_lock = inference_info[character]["queue_lock"]
    timeout = infer_para.get("timeout", 30)  # 超时时间（秒），默认 30 秒

    def inference_task():
        with torch.no_grad():
            geneface_log.debug(f"进入推理{infer_para['out_name']}")
            model.infer_once(infer_para) if is_parallel is None else inference_info[character]['model'].infer_once(infer_para)

    # with infer_lock:  # 保证队列操作线程安全
    if True:  # 模拟锁占位
        geneface_log.debug(f"AAA {character}开始推理{infer_queue.qsize()}, {infer_lock}")
        geneface_log.debug(f"BBB {character}开始推理{infer_queue.qsize()}")
        try:
            # 使用 ThreadPoolExecutor 执行推理任务
            future = thread_executor.submit(inference_task)
            future.result(timeout=timeout)  # 设置超时时间
            # with torch.no_grad():
            #     geneface_log.debug(f"进入推理{infer_para['out_name']}")
            #     model.infer_once(infer_para) if is_parallel is None else inference_info[character]['model'].infer_once(infer_para)
        # except concurrent.futures.TimeoutError as te:
        #     raise RuntimeError(f"{te}: Task for {character} timed out after {timeout} seconds!")
        # except Exception as e:
        #     raise RuntimeError(f"Error during inference: {e}")
        finally:
            torch_gc()
            if not infer_queue.empty():
                try:
                    geneface_log.debug(f"CCC {character}推理结束{infer_queue.qsize()}")
                    infer_queue.get()
                    geneface_log.debug(f"DDD {character}推理结束{infer_queue.qsize()}")
                    geneface_log.info(f"Task removed from queue for model: {character}")
                except Exception as e:
                    raise RuntimeError(f"Error cleaning up queue: {e}")


# def run_inference_sync(infer_para, blocking=True):
#     paste = infer_para['paste']
#     character = infer_para['character']
#     video_dir = infer_para['video_dir']
#     video_id = infer_para['video_id']
#     ori_video_path = inference_info[character]['ori_video']
#     crop_coordinates_tuple = inference_info[character]['crop_coordinates']
#     final_video_name = f"{video_id}.mp4"
#     cropped_video_name = f"{video_id}.cropped.mp4"
#     final_video_path = os.path.join(video_dir, final_video_name)
#     infer_video_path = os.path.join(video_dir, cropped_video_name)
#     logs = f"inference param: {infer_para}"
#     geneface_log.info(logs)
#     infer_para = copy.deepcopy(infer_para)
#     infer_para['drv_audio'] = infer_para['audio_path']
#     infer_para['out_name'] = final_video_path if paste == 'default' else infer_video_path
#     # 队列逻辑确保只有一个任务进行
#     model_path = inference_info[character]['torso_ckpt']
#     infer_queue = inference_info[character].get("queue", queue.Queue)
#     infer_lock = inference_info[character].get("lock", threading.Lock())
#     # queue.join()  # 等待队列完成当前任务
#     with infer_lock:  # 保证队列操作线程安全
#         if not infer_queue.empty():  # 如果队列不为空，说明已有任务在进行
#             msg = f'Model "{character}" is busy. Waiting for the current task to finish.'
#             geneface_log.info(msg)
#             infer_queue.join()  # 等待队列完成当前任务
#         msg = f"Adding task to model {character}'s queue."
#         geneface_log.info(msg)
#         geneface_log.info(f"AAA {character}开始推理{infer_queue.qsize()}, {infer_lock}")
#         infer_queue.put(video_id)
#         geneface_log.info(f"BBB {character}开始推理{infer_queue.qsize()}")
#         try:
#             # task = model.infer_once(infer_para)  # 进行推理
#             # infer_thread = asyncio.to_thread(model.infer_once, infer_para)
#             # task = asyncio.wait_for(infer_thread, timeout=60)  # 进行推理
#             # _ = await task if blocking else asyncio.create_task(task)
#             loop = asyncio.get_event_loop()
#             task = loop.run_in_executor(thread_executor, model.infer_once, infer_para)
#             _ = task if blocking else None
#         except asyncio.TimeoutError:
#             raise RuntimeError(f"Task for {character} timed out!")
#         finally:
#             torch_gc()
#             if not infer_queue.empty():
#                 try:
#                     geneface_log.info(f"CCC {character}推理结束{infer_queue.qsize()}")
#                     infer_queue.get()
#                     geneface_log.info(f"DDD {character}推理结束{infer_queue.qsize()}")
#                     infer_queue.task_done()
#                     geneface_log.info(f"Task removed from queue for model: {model_path}")
#                 except Exception as e:
#                     raise RuntimeError(f"Error cleaning up queue: {e}")
#
#     video_path = final_video_path if paste == 'default' else paste_back(infer_video_path, ori_video_path,
#                                                                         final_video_path, crop_coordinates_tuple)
#     return video_path


async def send_heartbeat(websocket):
    while True:
        await asyncio.sleep(30)  # 每30秒发送一次心跳
        await websocket.send_json({"type": "ping"})


# 初始化变量
config_data = read_json_file('config/config.json')
model: Optional[GeneFace2Infer] = None
storage_model_instances: Dict[str, Optional[GeneFace2Infer]] = {}  # {model_path: Model}
init_model_instances: Dict[str, GeneFace2Infer] = {}  # {model_path: Model}
# inference_info: Dict[str, Dict[str, Union[str, asyncio.Queue, asyncio.Lock, GeneFace2Infer]]] = config_data["inference_info"]
inference_info: Dict[str, Dict[str, Union[str, queue.Queue, threading.Lock, GeneFace2Infer]]] = config_data[
    "inference_info"]
base_audio_path = config_data['base_audio_path']
base_video_path = config_data['base_video_path']
host = config_data['host']
port = config_data['port']
secret_key = os.getenv('GENEFACE-SECRET-KEY', 'sk-geneface')
CURRENT_CHARACTER = "li"
parallel = False
thread_executor = ThreadPoolExecutor(max_workers=10)  # 设置线程池大小为 20
process_executor = ProcessPoolExecutor(max_workers=10)  # 设置线程池大小为 20
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
    logs = f"Inference request param: {request.model_dump()}"
    geneface_log.info(logs)
    sno = request.sno
    uid = request.uid
    blocking = request.blocking
    paste = request.paste
    character = request.character
    audio_name = request.audio_name
    is_parallel = request.parallel or parallel
    infer_para = get_arg(inference_info[character]['torso_ckpt'], inference_info[character]['head_ckpt'])
    inference_info[character]['infer_queue'].put(sno)
    video_dir = os.path.join(base_video_path, str(uid))
    os.makedirs(video_dir, exist_ok=True)
    identity = str(uuid.uuid4().hex[:8])
    audio_path = os.path.join(base_audio_path, audio_name)
    final_video_name = f"{identity}.mp4"
    cropped_video_name = f"{identity}.cropped.mp4"
    final_video_path = os.path.join(video_dir, final_video_name)
    infer_video_path = os.path.join(video_dir, cropped_video_name)
    infer_video_path = final_video_path if paste == 'default' else infer_video_path
    infer_data = {'identity': identity, 'character': character, 'blocking': blocking, 'paste': paste, "parallel": is_parallel,
                  'torso_ckpt': inference_info[character]['torso_ckpt'],
                  'head_ckpt': inference_info[character]['head_ckpt'],
                  'drv_audio': audio_path, 'out_name': infer_video_path, 'final_video_path': final_video_path}
    infer_para.update(infer_data)
    phase = 0
    try:
        if not os.path.exists(audio_path):
            msg = f'The audio file does not exist:{audio_path}!'
            raise FileNotFoundError(msg)
        _ = await refresh_models(infer_para) if is_parallel is None else await get_models(infer_para, is_parallel)
        # 如果需要阻塞以同步运行, 使用 asyncio.to_thread 使用线程异步运行同步推理函数, await等待完成
        # task = asyncio.wait_for(asyncio.to_thread(run_inference_sync, infer_data, blocking), timeout=300)
        # task = asyncio.wait_for(run_inference_async(infer_data), timeout=30)
        # infer_thread = threading.Thread(target=run_inference_sync, args=(infer_data,))
        # infer_thread.start()
        _ = await run_inference_async(infer_para)
        code = 0
        (phase, msg) = (2, "infer successful!") if blocking else (1, "infer starting...")
        video_data = [VideoData(video_dir=video_dir, video_name=f'{identity}.mp4')]
        result = VideoGenerateResponse(sno=sno, phase=phase, code=code, msg=msg, data=video_data)
        logs = f"Inference response results: {result.model_dump()}"
        geneface_log.info(logs)
        return JSONResponse(status_code=200, content=result.model_dump())
    except (json.JSONDecodeError, FileNotFoundError, asyncio.TimeoutError, RuntimeError, ValueError) as exception:
        # 定义一个映射关系，用于处理不同异常类型对应的 HTTP 状态码
        exception_to_status = {
            json.JSONDecodeError: 400,
            ValueError: 400,
            FileNotFoundError: 404,
            asyncio.TimeoutError: 500,
            RuntimeError: 500,
        }
        code = -1
        status_code = exception_to_status.get(type(exception), 500)
        msg = f"{type(exception).__name__}: {str(exception)}"
        error_message = VideoRetrieveResponse(
            code=code,
            phase=phase,
            msg=msg,
        )
        logs = f"Retrieve response error: {error_message}"
        geneface_log.error(logs)
        return JSONResponse(status_code=status_code, content=error_message.model_dump())


@geneface_app.post("/v1/video/retrieve")
@geneface_app.post("/v1/video/get")
async def retrieve_video(request: VideoRetrieveRequest):
    phase = 2
    try:
        logs = f"Retrieve request param: {request.model_dump()}"
        geneface_log.info(logs)
        sno = request.sno
        uid = request.uid
        video_num = request.video_num
        video_dir = os.path.join(base_video_path, str(uid))
        video_files = [file for file in os.listdir(video_dir) if
                       file.endswith('.mp4') and not file.endswith(('.tmp.mp4', '.crop.mp4'))]
        sorted_files = sorted(video_files, key=lambda x: os.path.getmtime(os.path.join(video_dir, x)), reverse=True)[
                       :video_num]
        phase = 2
        (code, msg) = (0, f"Successfully retrieved {len(sorted_files)} video files") if sorted_files else (
            1, "No videos found")
        video_data = [VideoData(video_name=video_name, video_dir=video_dir) for video_name in sorted_files]
        result = VideoRetrieveResponse(
            sno=sno,
            phase=phase,
            code=code,
            msg=msg,
            data=video_data)
        logs = f"Retrieve response results: {result.model_dump()}"
        geneface_log.info(logs)
        return JSONResponse(status_code=200, content=result.model_dump())
    except (json.JSONDecodeError, FileNotFoundError, asyncio.TimeoutError, RuntimeError, ValueError) as exception:
        # 定义一个映射关系，用于处理不同异常类型对应的 HTTP 状态码
        exception_to_status = {
            json.JSONDecodeError: 400,
            ValueError: 400,
            FileNotFoundError: 404,
            asyncio.TimeoutError: 500,
            RuntimeError: 500,
        }
        code = -1
        status_code = exception_to_status.get(type(exception), 500)
        msg = f"{type(exception).__name__}: {str(exception)}"
        error_message = VideoRetrieveResponse(
            code=code,
            phase=phase,
            msg=msg,
        )
        logs = f"Retrieve response error: {error_message}"
        geneface_log.error(logs)
        return JSONResponse(status_code=status_code, content=error_message.model_dump())


# 定义 WebSocket 路由
@geneface_app.websocket("/v1/video/generate")
@geneface_app.websocket("/v1/video/inference")
async def websocket_endpoint(websocket: WebSocket):
    client_host = websocket.client.host
    client_port = websocket.client.port
    # 记录客户端连接信息
    connect_msg = f"WebSocket connected from {client_host}:{client_port}"
    geneface_log.info(connect_msg)
    # 接受连接
    await websocket.accept()
    global CURRENT_CHARACTER
    try:
        while True:
            phase = 0
            # 接收客户端发送的数据
            data = await websocket.receive_json()  # 自动解析接收到的 JSON 数据并将其转换为字典
            try:
                request = VideoGenerateRequest.model_validate_json(data)
                logs = f"Inference request param: {request.model_dump()}"
                geneface_log.info(logs)
                sno = request.sno
                uid = request.uid
                blocking = request.blocking
                paste = request.paste
                character = request.character
                audio_name = request.audio_name
                is_parallel = request.parallel or parallel
                infer_para = get_arg(inference_info[character]['torso_ckpt'], inference_info[character]['head_ckpt'])
                inference_info[character]['infer_queue'].put(sno)
                video_dir = os.path.join(base_video_path, str(uid))
                os.makedirs(video_dir, exist_ok=True)
                video_id = str(uuid.uuid4().hex[:8])
                audio_path = os.path.join(base_audio_path, audio_name)
                final_video_name = f"{video_id}.mp4"
                cropped_video_name = f"{video_id}.cropped.mp4"
                final_video_path = os.path.join(video_dir, final_video_name)
                infer_video_path = os.path.join(video_dir, cropped_video_name)
                infer_video_path = final_video_path if paste == 'default' else infer_video_path
                infer_data = {'character': character, 'blocking': blocking, 'paste': paste,
                              'torso_ckpt': inference_info[character]['torso_ckpt'],
                              'head_ckpt': inference_info[character]['head_ckpt'],
                              'drv_audio': audio_path, 'out_name': infer_video_path,
                              'final_video_path': final_video_path}
                infer_para.update(infer_data)
                if not os.path.exists(audio_path):
                    msg = f'The audio file does not exist:{audio_path}!'
                    raise FileNotFoundError(msg)

                # 刷新或加载模型
                _ = await refresh_models(infer_para) if is_parallel is None else await get_models(infer_para, is_parallel)
                # 返回推理开始的消息
                msg = "Inference starting..."
                code = 0
                phase = 1
                video_data = [VideoData(video_dir=video_dir, video_name=f'{video_id}.mp4')]
                result = VideoGenerateResponse(sno=sno, phase=phase, code=code, msg=msg, data=video_data)
                logs = f"Inference response results: {result.model_dump()}"
                geneface_log.info(logs)
                await websocket.send_json(result.model_dump())
                infer_data = {"uid": uid, "character": character, "paste": paste,
                              "audio_path": audio_path, "video_dir": video_dir, "video_id": video_id}
                # # 启动推理线程
                # infer_thread = threading.Thread(target=run_inference_sync, args=(infer_data,))
                # infer_thread.start()
                # 使用 asyncio.to_thread 调用推理函数
                # await asyncio.wait_for(run_inference_async(infer_data), timeout=30)
                # await asyncio.wait_for(asyncio.to_thread(run_inference_sync, infer_data), timeout=30)
                _ = await run_inference_async(infer_para)
                msg = "Inference success"
                code = 0
                phase = 2
                video_data = [VideoData(video_dir=video_dir, video_name=f'{video_id}.mp4')]
                result = VideoRetrieveResponse(
                    sno=sno,
                    phase=phase,
                    code=code,
                    msg=msg,
                    data=video_data)
                logs = f"Retrieve response results: {result.model_dump()}"
                geneface_log.info(logs)
                await websocket.send_json(result.model_dump())
            except (json.JSONDecodeError, ValidationError, FileNotFoundError, ValueError, RuntimeError, Exception) as e:
                code = -1
                error_type = type(e).__name__
                msg = f'{error_type}: {e}'
                error_message = VideoRetrieveResponse(
                    code=code,
                    phase=phase,
                    msg=msg,
                )
                logs = f"Retrieve response error: {error_message}"
                geneface_log.error(logs)
                await websocket.send_json(error_message)
    except WebSocketDisconnect:
        msg = f"WebSocket disconnected from {client_host}:{client_port}"
        geneface_log.info(msg)
        # 如果需要发送消息或执行额外清理操作，可以在这里添加
        # await websocket.close()  # 确保 WebSocket 已正确关闭


if __name__ == "__main__":
    asyncio.run(init_app())
    # uvicorn.run(geneface_app, host=host, port=8042)
    uvicorn.run(geneface_app, host=host, port=int(port))
