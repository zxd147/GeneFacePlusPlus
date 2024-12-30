import json
import os
import queue
import socket
import time

from flask import Flask, request
from flask_cors import CORS

from utils.uitls import read_json_file


app = Flask(__name__)
CORS(app)

remote_dict = {}
ip_mapping = {}
text_mapping = {}
SNO_START = 123
data_json = read_json_file('config/config.json')
base_video_path = data_json['base_video_path']
base_audio_path = data_json['base_audio_path']
socket_host = data_json['socket_ip_host']
socket_port = data_json['socket_ip_port']
stream_url = data_json['rtsp_url']
socket_ip = data_json['socket_ip_host']
access_key = data_json['access_key']
secret_key = data_json['secret_key']
upload_server = data_json['api_server_name']
message_queue = queue.Queue()
inference_queue = queue.Queue()
isBusy = False
gWsk_remote = None


def send_data_to_modeler(inference_info):
    # 创建一个新的套接字对象, AF_INET 表示使用 IPv4 地址, SOCK_STREAM 表示这是一个流式套接字（TCP）
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(2)  # 设置2s超时时间
    cur_id = inference_info['session_id']
    os.makedirs(os.path.join(base_video_path, str(cur_id)), exist_ok=True)
    print(f"{time.ctime()} 当前发送的推理信息 {inference_info}")  # 'time.ctime(): Wed Oct  9 17:04:28 2024'

    try:
        client_socket.connect((socket_host, int(socket_port)))
        # inference_info = str(inference_info)
        # client_socket.send(inference_info.encode("utf-8"))
        # 将字典转换为 JSON 字符串并发送
        # 调用 .encode("utf-8") 将字符串编码为字节。这是因为网络通信需要发送字节数据，而不是字符串。
        inference_info_json = json.dumps(inference_info)  # 转换为 JSON 字符串
        client_socket.send(inference_info_json.encode("utf-8"))  # 编码为字节
        # 接收服务器的响应, recv(1024) 方法会阻塞，直到接收到数据，或者连接关闭。
        return_data = client_socket.recv(1024).decode('utf-8')
        if return_data:
            print("接受socket返回", return_data)
    except socket.timeout:
        print("连接超时")
    client_socket.close()


@app.route("/inference", methods=['POST'])
def receive_data():
    global remote_dict
    request_data = request.get_json()
    audio_name = request_data['audio_name']  # 只需要XXX.WAV 即可
    project_type = request_data['project_type']
    session_id = request_data['uid']

    if session_id == "dentist":
        remote_dict.setdefault(session_id, request.remote_addr)
        print(f"remote = {remote_dict[session_id]}" if
              remote_dict[session_id] == request.remote_addr else "remote 已保存！")
    wav_path = os.path.join(base_audio_path, audio_name)
    if os.path.exists(wav_path):
        status = 3
        ip_mapping.setdefault(session_id, len(ip_mapping) + 1)
        info = {"session_id": session_id, 'audio': audio_name, "project_type": project_type, }  # 构建ip信息库 ，并将信息传入到队列
        send_data_to_modeler(info)
    else:
        status = 4
    return {"status": status, "id": session_id, "timestamp": int(time.time()), }


@app.route('/live/inference', methods=['POST'])
def live_info():  # 数字人后台的接口
    inference_data = request.get_json()
    source_video = inference_data.get("person_id")
    driven_audio = inference_data.get("driven_audio")
    data_type = inference_data.get("data_type")
    task_id = inference_data.get("task_id")
    wav_path = os.path.join(base_audio_path, driven_audio)
    msg = f'got audio file: {wav_path}' if os.path.exists(wav_path) else f'audio file not found: {wav_path}'
    # 推理的进行没写
    result = {
        "status": 4,
        "msg": f"empty interface status always equal 4, {msg}",
        "timestamp": int(time.time())
    }
    return result


@app.route("/get_video", methods=['POST'])
def return_files_list():
    global SNO_START
    data_json = request.get_json()  # 会话交互
    if 'uid' not in data_json:
        return {"status": 4, "timestamp": int(time.time())}
    ask_id = data_json['uid']
    sno_msg = SNO_START = data_json.setdefault('sno', SNO_START + 1)
    video_path = os.path.join(base_video_path, str(ask_id))
    if not os.path.exists(video_path):
        return {"status": 5, "timestamp": int(time.time()), "message": video_path}
    files = os.listdir(video_path)
    sound_files = [os.path.join(video_path, file) for file in files if file.endswith('.mp4')]
    sorted_files = sorted(sound_files, key=lambda x: os.path.getmtime(x), reverse=True)
    sorted_files_top5 = sorted_files[:5]
    return_data = {
        "stream_url": stream_url,
        "timestamp": int(time.time()),
        "session": ask_id,
        "data": [],
        "sno": sno_msg
    }

    if len(sorted_files_top5) == 0:
        return return_data
    for i, filename in enumerate(sorted_files_top5):
        if filename.split(".")[-2] != 'tmp':
            new_path = filename.replace(base_video_path, "video")  # 20240314 这里换路径有点太随意了，要注意
            return_data['data'].append({
                "video_name": new_path,
                "time": int(os.path.getctime(filename)),
                "uid": ask_id
            })
    return return_data


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5463, debug=False)


