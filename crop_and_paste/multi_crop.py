import argparse
import json
import os
import subprocess

import cv2
import face_alignment
import numpy as np
import torch
from moviepy.editor import VideoFileClip, ImageSequenceClip
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def multi_crop(input_path, output_path=None, resized=True, crop_size=960, target_size=512, target_fps=25):
    base_name = input_path.split(".")[0].split("_")[0].split("-")[0]
    base_dir = os.path.dirname(input_path)

    video_width, video_height = get_video_resolution(input_path)
    print(f"Original Width: {video_height}, Original Height: {video_height}")
    aspect_ratio = video_height / video_width
    resize_ratio = target_size / crop_size
    resize_width = (int(video_width * resize_ratio + 1)) // 2 * 2
    resize_height = (int(resize_width * aspect_ratio) + 1) // 2 * 2

    # 转分辨率后的视频路径
    resized_path = os.path.join(base_dir, f'{base_name}_resized.mp4')
    #  裁剪的音频的输出
    audio_path = os.path.join(base_dir, f'{base_name}_audio.wav')
    # 最终生成的视频路径
    final_path = output_path or os.path.join(base_dir, f'{base_name}.mp4')
    final_path = os.path.join(base_dir, base_name, '_crop', '.mp4') if os.path.isdir(final_path) else final_path
    # 保存人脸特征坐标的文件
    landmarks_path = os.path.join(base_dir, f"{base_name}.npy")
    #  最终生成的视频帧的路径
    frames_dir = os.path.join(base_dir, f'{base_name}_frames')

    if resized:
        resize_video(input_path, resized_path, target_fps, resize_width, resize_height)

    crop_video_by_cv2(input_path, frames_dir, landmarks_path, target_size, target_fps=25)
    audio = extract_audio_by_moviepy(input_path, audio_path)
    compose_video_by_moviepy(base_dir, base_name, frames_dir, audio, final_path, target_fps)
    print('multi crop done!')


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
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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


def sort_key(filename):
    # Split the filename into name and extension
    name, ext = os.path.splitext(filename)
    # Split the name into prefix and number
    prefix, number = name.rsplit('_', 1)
    # Convert the number to an integer and return it
    return int(number)


def crop_video_by_cv2(input_path, frames_dir, target_size, landmarks_path, target_fps=25):
    os.makedirs(frames_dir, exist_ok=True)
    # Load the video
    video_capture = cv2.VideoCapture(input_path)
    video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # Initialize the face alignment pipeline
    face_ali = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device, flip_input=False)
    video_fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    # Calculate the interval to pick frames
    interval = max(1, video_fps // target_fps)

    # #  根据第一帧的人脸坐标信息,逐帧进行裁剪
    # frame_number = 0
    # print('Start cropping video...')
    # # 初始化 tqdm 进度条
    # with tqdm(total=total_frames, desc="Cropping frames") as pbar:
    #     while video_capture.isOpened():
    #         is_ret, frame = video_capture.read()
    #         if not is_ret:
    #             break
    #
    #         # 裁剪图像并保存
    #         cropped_image = frame[start_y:start_y + target_size, start_x:start_x + target_size]
    #         frame_name = f'frame_{frame_number:04d}.png'
    #         frame_path = os.path.join(frames_dir, frame_name)
    #         cv2.imwrite(frame_path, cropped_image)
    #         # 更新 tqdm 进度条
    #         pbar.update(1)
    #         frame_number += 1
    # # 释放视频资源
    # video_capture.release()

    landmarks_list = []
    frame_index = 0
    with tqdm(total=total_frames, desc="Processing video frames", unit="frame") as pbar:
        while frame_index < total_frames:
            # Set the video position to the current frame index
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            is_success, frame = video_capture.read()
            if not is_success:
                print(f"Error reading frame {frame_index}")
                frame_index += interval
                continue

            # Detect landmarks
            landmarks = face_ali.get_landmarks(frame)
            if not landmarks:
                print(f"No faces detected in frame {frame_index}")
                frame_index += interval
                continue

            # Get the 30th landmark (index 29), 获取第一张脸(索引为0)的第30个特征点(索引为29, 通常表示鼻尖位置）
            landmark_30 = landmarks[0][29]
            landmarks_list.append(landmark_30)
            half_size = (target_size + 1) // 2
            # Calculate the crop region
            center_x, center_y = int(landmark_30[0]), int(landmark_30[1])

            x1, y1 = max(0, center_x - half_size), max(0, center_y - half_size)
            x2, y2 = x1 + target_size, y1 + target_size
            # Ensure the crop region is within the frame boundaries
            if x2 > frame.shape[1]:
                x2 = frame.shape[1]
                x1 = x2 - target_size
            if y2 > frame.shape[0]:
                y2 = frame.shape[0]
                y1 = y2 - target_size

            cropped_frame = frame[y1:y2, x1:x2]
            # Save the cropped frame
            output_image_path = os.path.join(frames_dir, f'cropped_frame_{frame_index}.jpg')
            cv2.imwrite(output_image_path, cropped_frame)
            # 更新进度条
            frame_index += interval
            pbar.update(interval)
    # Save landmarks to a file
    np.save(landmarks_path, np.array(landmarks_list))
    print(f"Landmarks saved to {landmarks_path}")
    video_capture.release()


def extract_audio_by_moviepy(resized_path, audio_path):
    # Extract audio from the original video
    print(f"Starting extract audio...")
    video = VideoFileClip(resized_path)
    audio = video.audio
    audio.write_audiofile(audio_path)
    print(f"Extract audio successful. Output audio saved to {audio_path}")
    return audio


def compose_video_by_moviepy(base_dir, base_name, frames_dir, audio, final_path, target_fps):
    # Get a list of all image files in the folder
    image_files = os.listdir(frames_dir)
    # Sort the list in the order(顺序) you want
    image_files.sort(key=sort_key)
    # Create full paths for each image file
    image_paths = [os.path.join(frames_dir, filename) for filename in image_files]

    # Create a video from the image sequence
    video_without_audio = ImageSequenceClip(image_paths, fps=target_fps)
    video_without_audio.write_videofile(os.path.join(base_dir, f"{base_name}_no_audio.mp4"))
    # Add audio to the video
    video_with_audio = video_without_audio.set_audio(audio)
    video_with_audio.write_videofile(os.path.join(base_dir, f"{base_name}.mp4"), codec='libx264')


def compose_video_by_ffmpeg(frames_dir, audio_path, final_path, target_fps):
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
    # 运行命令
    try:
        print(f"Starting compose videos...")
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True, text=True)
        print(f"Compose videos successful. Output video saved to {final_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while composing videos: {e}")


if __name__ == '__main__':
    default_in_path = '/home/zxd/code/Vision/GeneFacePlusPlus/data/raw/videos/yu/yu_ori.mp4'
    # default_out_path = '/home/zxd/code/Vision/GeneFacePlusPlus/data/raw/videos/yu/yu.mp4'
    default_out_path = default_in_path.replace("_ori", '_crop')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=default_in_path)  # 输入的视频路径
    parser.add_argument('--output', type=str, default=default_out_path)  # 输出的视频路径
    args = parser.parse_args()
    multi_crop(args.input, args.output, resized=True, crop_size=960, target_size=512, target_fps=25)


