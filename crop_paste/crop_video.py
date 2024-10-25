import json

import cv2
import subprocess
import dlib
import os
import argparse
import numpy as np


def crop(input_video):
    width, height = get_video_resolution(input_video)
    print(f"Original Width: {width}, Original Height: {height}")
    aspect_ratio = width / height
    resize_width = 720
    resize_height = int(resize_width / aspect_ratio)
    target_size = 512
    target_fps = 25

    # 使用 os.path.basename 获取视频文件名（不包含扩展名）
    name = os.path.basename(input_video).split('.')[0].replace('_ori', '')
    # 使用 os.path.dirname 获取视频文件所在的目录路径
    base_dir = os.path.dirname(input_video)
    # 转分辨率后的视频路径
    resized_path = os.path.join(base_dir, f'{name}_resized.mp4')
    # 第一帧保存路径
    first_frame_face_path = os.path.join(base_dir, f'{name}_first_frame_face.png')
    #  裁剪的音频的输出
    audio_path = os.path.join(base_dir, f'{name}_audio.wav')
    #  最终生成的视频帧的路径
    frames_dir = os.path.join(base_dir, f'{name}_frames')
    # 最终生成的视频路径
    final_path = os.path.join(base_dir, f'{name}.mp4')
    # 保存人脸特征坐标的文件
    landmarks_file = os.path.join(base_dir, f"{name}.npy")

    # 初始化dlib的人脸检测器
    face_detector = dlib.get_frontal_face_detector()
    resize_video(input_video, resized_path, target_fps, resize_width, resize_height)
    video_capture = cv2.VideoCapture(resized_path)

    diy_number = 1
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        print(f"第{diy_number}帧的大小", frame.shape)
        face_coords = get_face_coordinates(face_detector, frame, first_frame_face_path)
        if face_coords is not None:
            center_x, center_y = face_coords
            print(f"Center coordinates of the {diy_number} frame detected face:", center_x, center_y)
        else:
            print("No face detected in the first frame, will exit.")
            exit(1)
        diy_number += 1
    # 获取第一帧的人脸坐标
    ret, first_frame = video_capture.read()
    if not ret:
        print("Failed to read the first frame.")
        video_capture.release()
        exit(1)
    face_coords = get_face_coordinates(face_detector, first_frame, first_frame_face_path)
    center_x, center_y = face_coords
    landmarks = [center_x, center_y, target_size, target_size]
    # Save landmarks to a file
    np.save(landmarks_file, np.array(landmarks))
    print(f"Landmarks saved to {landmarks_file}")

    # extract_audio(resized_path, audio_path)
    # crop_video(frames_dir, target_size, center_x, center_y, video_capture)
    # compose_videos(frames_dir, audio_path, target_fps, final_path)
    crop_compose_video(resized_path, final_path, target_size, center_x, center_y)
    print('crop done!')


def get_video_resolution(input_video):
    # 使用 FFmpeg 获取视频信息
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'json'
    ]
    result = subprocess.run(cmd + [input_video], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    data = json.loads(result.stdout.decode())
    width = data['streams'][0]['width']
    height = data['streams'][0]['height']
    return width, height


def resize_video(input_video, resized_path, target_fps, resize_width, resize_height):
    # 定义 FFmpeg 命令
    command = [
        "ffmpeg",
        "-i", input_video,
        "-vf", f"fps={target_fps},scale=w={resize_width}:h={resize_height}",
        "-qmin", "1",
        "-q:v", "1",
        "-y", resized_path
    ]
    # 运行命令
    try:
        print(f"Starting resized video to {resize_width}x{resize_height}...")
        subprocess.run(command, check=True)
        print(f"Conversion successful. Output file saved to {resized_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while converting the video: {e}")


def extract_audio(resized_path, audio_path):
    # 提取视频的音频
    command = [
        "ffmpeg",
        "-i", resized_path,
        '-f', 'wav',
        '-ar', '16000',
        '-y', audio_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # 运行命令
    try:
        print(f"Starting extract audio...")
        subprocess.run(command, check=True)
        print(f"Extract audio successful. Output audio saved to {audio_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while extracting the  audio: {e}")


def get_face_coordinates(face_detector, image, first_frame_face_path):
    # cv读取的图片转为RGB格式
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 使用dlib的人脸检测器检测人脸
    detections = face_detector(rgb_image)
    if len(detections) > 0:
        face = detections[0]
        # 计算人脸边界框的坐标
        face_x1 = face.left()
        face_y1 = face.top()
        face_x2 = face.right()
        face_y2 = face.bottom()
        # 确保裁剪区域在图像范围内
        face_x1 = max(face_x1, 0)
        face_y1 = max(face_y1, 0)
        face_x2 = min(face_x2, image.shape[1])
        face_y2 = min(face_y2, image.shape[0])
        # 裁剪图像
        print(f'image[{face_y1}:{face_y2}, {face_x1}:{face_x2}]')
        crop_face = image[face_y1:face_y2, face_x1:face_x2]
        # 保存裁剪后的人脸图像
        cv2.imwrite(first_frame_face_path, crop_face)
        # 计算并返回人脸中心点
        face_x = (face_x1 + face_x2) // 2
        face_y = (face_y1 + face_y2) // 2
        print("Center coordinates of the first detected face:", face_x, face_y)
        return face_x, face_y
    else:
        print("No face detected in the first frame, will exit.")
        exit(1)


def crop_video(frames_dir, target_size, center_x, center_y, video_capture):
    os.makedirs(frames_dir, exist_ok=True)
    crop_size = target_size // 2
    start_x = max(center_x - crop_size, 0)
    start_y = max(center_y - crop_size, 0)
    #  根据第一帧的人脸坐标信息,逐帧进行裁剪
    frame_number = 0
    print('Start cropping video...')
    # 获取视频总帧数
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    while video_capture.isOpened():
        is_ret, frame = video_capture.read()
        if not is_ret:
            break
        cropped_image = frame[start_y:start_y + target_size, start_x:start_x + target_size]
        frame_name = f'frame_{frame_number:04d}.png'
        frame_path = os.path.join(frames_dir, frame_name)
        cv2.imwrite(frame_path, cropped_image)
        # 输出当前进度
        print(f'Processing frame {frame_number}/{total_frames}...')
        frame_number += 1
    video_capture.release()


def compose_videos(frames_dir, audio_path, target_fps, final_path):
    frame_pattern = f"{frames_dir}/frame_%04d.png"
    #  把裁剪人脸后的视频帧和音频无损合并为 最终的视频
    command = [
        "ffmpeg",
        "-i", frame_pattern,
        "-i", audio_path,
        "-c:v", "libx264",
        "-framerate", str(target_fps),
        '-pix_fmt', 'yuv420p',
        "-y", final_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # 运行命令
    try:
        print(f"Starting compose videos...")
        subprocess.run(command, check=True)
        print(f"Compose videos successful. Output video saved to {final_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while composing videos: {e}")


def crop_compose_video(resized_path, final_path, target_size, center_x, center_y):
    crop_size = target_size // 2
    start_x = max(center_x - crop_size, 0)
    start_y = max(center_y - crop_size, 0)
    print(f'裁剪的左上角坐标为: [{start_x}, {start_y}]')
    command = [
        "ffmpeg",
        "-i", resized_path,
        "-vf", f"crop={target_size}:{target_size}:{start_x}:{start_y}",  # 左上角坐标, 而且是从左上角开始计算, 即第四象限
        "-qmin", "1",
        "-q:v", "1",
        "-y", final_path
    ]
    print('command: ', command)
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # 运行命令
    try:
        print(f"Starting crop videos...")
        subprocess.run(command, check=True)
        print(f"Crop videos successful. Output video saved to {final_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while cropping videos: {e}")


if __name__ == "__main__":
    args = parser = argparse.ArgumentParser()
    default_video = '/home/zxd/code/Vision/GeneFacePlusPlus/temp/li_logo_trimmed_1440x2560.mp4'
    parser.add_argument('--input', type=str, default=default_video)
    opt = parser.parse_args()
    input_path = opt.input
    crop(input_path)

