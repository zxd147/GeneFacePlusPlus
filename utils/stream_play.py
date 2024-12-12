"""
单纯使用FFMpeg来跑，但是可持续的播放不理想
"""

import subprocess
import json
import time
import os

with open("helper/config.json", 'r', encoding='UTF-8') as f:
    data_json = json.load(f)


def start_publish(video_file, rtsp_url):
    # if os.path.exists(video_file):
    #     # 需要做2s的空白画面
    #     from moviepy import
    # FFmpeg 推流命令
    ffmpeg_cmd = [
        'ffmpeg',
        '-re',
        '-i', video_file if video_file else data_json['video_file_path'],
        '-c', 'copy',
        '-f', 'rtsp',
        rtsp_url or data_json['rtsp_url']
    ]

    print(f'{time.ctime()}  查看ffmpeg,{ffmpeg_cmd}')

    # 启动 FFmpeg 推流进程
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # # 等待 FFmpeg 进程结束
    stdout, stderr = ffmpeg_proc.communicate()
    recode = ffmpeg_proc.returncode
    # 检查 FFmpeg 进程的退出状态
    if recode == 0:
        print('FFmpeg push completed successfully.')
    else:
        print('FFmpeg push failed with return code:', recode)
        print('FFmpeg stdout:', stdout.decode())
        print('FFmpeg stderr:', stderr.decode())
