import subprocess
import time
from datetime import datetime

import cv2


def get_video_info_by_cv2(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    return duration, fps, frame_count


def get_video_info_by_ffmpeg(video_path):
    """一次性获取视频的时长、帧率和总帧数"""
    start = time.time()
    command = [
        'ffprobe',
        '-v', 'error',  # 禁止输出不必要的信息
        '-select_streams', 'v:0',  # 选择视频流
        '-show_entries', 'stream=r_frame_rate,duration,nb_frames',  # 获取时长、帧率和总帧数
        '-of', 'default=noprint_wrappers=1:nokey=1',  # 输出格式设置
        video_path
    ]
    # 运行命令
    try:
        result = subprocess.run(command, capture_output=True, check=True, text=True)
        output = result.stdout.strip().split('\n')
        fps = eval(output[0])  # 帧率可能是 "numerator/denominator" 格式，需要转换为浮动数值
        duration = float(output[1])  # 时长
        frame_count = int(output[2])   # 获取总帧数
        return duration, fps, frame_count
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while getting video info: {e}")
        return None, None, None
    finally:
        end = time.time()
        print(f'Elapsed time: {end - start} seconds.')


if __name__ == '__main__':
    input_video_path = 'temp/20241224/gu_161350.mp4'

    start_time = datetime.now()
    video_duration, video_fps, video_frame = get_video_info_by_ffmpeg(input_video_path)
    print(f"Duration: {video_duration} seconds")
    print(f"FPS: {video_fps}")
    print(f"Frame count: {video_frame}")
    end_time = datetime.now()
    print(f'{(end_time - start_time).total_seconds()} s')

    start_time = cv2.getTickCount()
    video_duration, video_fps, video_frame = get_video_info_by_cv2(input_video_path)
    print(f"Duration: {video_duration} seconds")
    print(f"FPS: {video_fps}")
    print(f"Frame count: {video_frame}")
    end_time = cv2.getTickCount()
    print(f'{(end_time - start_time) / cv2.getTickFrequency()} s')


