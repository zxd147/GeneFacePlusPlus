import argparse
import json
import os
import subprocess

import cv2
import dlib
import numpy as np


def crop_out(input_path, output_path=None, resized=True, crop_size=960, target_size=512, target_fps=25, index=1):
    base_name = input_path.sprit(".")[0]("_")[0]("-")[0]
    base_dir = os.path.dirname(input_path)
    # 初始化dlib的人脸检测器
    face_detector = dlib.get_frontal_face_detector()
    
    video_width, video_height = get_video_resolution(input_path)
    print(f"Original Width: {video_height}, Original Height: {video_height}")
    aspect_ratio = video_height / video_width
    resize_ratio = target_size / crop_size
    resize_width = (int(video_width * resize_ratio + 1)) // 2 * 2
    resize_height = (int(resize_width * aspect_ratio) + 1) // 2 * 2

    # 转分辨率后的视频路径
    resized_path = os.path.join(base_dir, f'{base_name}_resized.mp4')
    # 第一帧保存路径
    face_path = os.path.join(base_dir, f'{base_name}_face_frame.png')
    #  裁剪的音频的输出
    audio_path = os.path.join(base_dir, f'{base_name}_audio.wav')
    # 最终生成的视频路径
    final_path = output_path or os.path.join(base_dir, f'{base_name}.mp4')
    final_path = os.path.join(base_dir, base_name, '_crop', '.mp4') if os.path.isdir(final_path) else final_path
    # 保存人脸特征坐标的文件
    landmarks_path = os.path.join(base_dir, f"{base_name}.npy")
    
    if resized:
        resize_video(input_path, resized_path, target_fps, resize_width, resize_height)    

    face_coords = None
    frame_number = 1
    video_capture = cv2.VideoCapture(resized_path)
    video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while video_capture.isOpened():
        if frame_number == index:
            ret, frame = video_capture.read()
            if not ret:
                video_capture.release()
                raise ValueError(f"第{frame_number}帧无法读取视频文件。")
            # 获取人脸
            face_coords = detect_face_center(face_detector, frame, face_path)
            center_x, center_y = face_coords
            print(f"第{frame_number}帧检测到人脸中心坐标: ({center_x}, {center_y})")
            break
        frame_number += 1

    if not face_coords:
        video_capture.release()
        raise ValueError(f"第{index}帧未检测到人脸")
    video_capture.release()
    
    center_x, center_y = face_coords
    # 计算裁剪区域的起始点, +1为了向上取整
    start_x = center_x - (target_size + 1) // 2
    start_y = center_y - (target_size + 1) // 2
    # 限制裁剪区域的起始点，确保裁剪区域不会超出视频边界
    start_x = int(max(0, min(start_x, video_width - target_size)))
    start_y = int(max(0, min(start_y, video_height - target_size)))
    landmarks = [start_x, start_y, target_size, target_size]
    # Save landmarks to a file
    np.save(landmarks_path, np.array(landmarks))
    print(f"Landmarks saved to {landmarks_path}")

    extract_audio_by_ffmpeg(resized_path, audio_path)
    crop_compose_video_by_ffmpeg(resized_path, final_path, target_size, target_fps, start_x, start_y)
    print('crop done!')


def get_video_resolution(input_video):
    # 使用 FFmpeg 获取视频信息
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'json'
    ]
    command = command + [input_video]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    data = json.loads(result.stdout.decode())
    width = data['streams'][0]['width']
    height = data['streams'][0]['height']
    return width, height


def resize_video(input_video, resized_path, target_fps, resize_width, resize_height):
    # 定义 FFmpeg 命令
    command = [
        "ffmpeg",
        "-i", input_video,
        "-vf", f"scale=w={resize_width}:h={resize_height}, fps={target_fps}",
        "-qmin", "1",
        "-q:v", "1",
        "-y", resized_path
    ]
    # 运行命令
    try:
        print(f"Starting resized video to {resize_width}x{resize_height}...")
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True, text=True)
        print(f"Conversion successful. Output file saved to {resized_path}")
        print(f"The scaled resolution is: {resize_width}x{resize_height}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while converting the video: {e}")
        

def extract_audio_by_ffmpeg(resized_path, audio_path):
    # 提取视频的音频
    command = [
        "ffmpeg",
        "-i", resized_path,
        '-f', 'wav',
        '-ar', '16000',
        '-y', audio_path
    ]
    # 运行命令
    try:
        print(f"Starting extract audio...")
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True, text=True)
        print(f"Extract audio successful. Output audio saved to {audio_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while extracting the  audio: {e}")


def detect_face_center(face_detector, image, face_frame_path):
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
        print(f'face frame: image[{face_y1}:{face_y2}, {face_x1}:{face_x2}]')
        crop_face = image[face_y1:face_y2, face_x1:face_x2]
        if face_frame_path:
            # 保存裁剪后的人脸图像
            cv2.imwrite(face_frame_path, crop_face)
        # 计算并返回人脸中心点
        face_x = (face_x1 + face_x2) // 2
        face_y = (face_y1 + face_y2) // 2
        print("Center coordinates of the first detected face:", face_x, face_y)
        return face_x, face_y
    else:
        print("No face detected in the first frame, will exit.")
        exit(1)


def crop_compose_video_by_ffmpeg(resized_path, final_path, target_size, target_fps, start_x, start_y):
    command = [
        "ffmpeg",
        "-i", resized_path,
        "-vf", f"crop={target_size}:{target_size}:{start_x}:{start_y}, fps={target_fps}",  # 左上角坐标, 而且是从左上角开始计算, 即第四象限
        "-qmin", "1",
        "-q:v", "1",
        "-y", final_path
    ]
    print('command: ', command)
    print(f'裁剪的左上角坐标为: [{start_x}, {start_y}]')
    # 运行命令
    try:
        print(f"Starting crop videos...")
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True, text=True)
        print(f"Crop videos successful. Output video saved to {final_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while cropping videos: {e}")


if __name__ == "__main__":
    default_in_path = '/home/zxd/code/Vision/GeneFacePlusPlus/data/raw/videos/yu_ori.mp4'
    default_out_path = default_in_path.replace("_ori", '_crop')
    args = parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=default_in_path)
    parser.add_argument('--output', type=str, default=default_out_path)
    opt = parser.parse_args()
    input_video_path = opt.input
    output_video_path = opt.input
    crop_out(input_video_path, output_video_path, resized=True, crop_size=600, target_size=512, target_fps=25, index=1)
    
    
