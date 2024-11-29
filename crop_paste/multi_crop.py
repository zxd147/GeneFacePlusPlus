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


def resize_video(input_video, target_fps, crop_size, target_size=512):
    resized_path = input_video.replace("_ori", "_resized")
    width, height = get_video_resolution(input_video)
    print(f"Original Width: {width}, Original Height: {height}")
    aspect_ratio = height / width
    crop_ratio = target_size / crop_size
    resize_width = width * crop_ratio // 2 * 2
    resize_height = int(resize_width * aspect_ratio) // 2 * 2
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
        return resized_path
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while converting the video: {e}")


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


def sort_key(filename):
    # Split the filename into name and extension
    name, ext = os.path.splitext(filename)
    # Split the name into prefix and number
    prefix, number = name.rsplit('_', 1)
    # Convert the number to an integer and return it
    return int(number)


def save_landmarks(landmarks, output_landmarks_path):
    np.save(output_landmarks_path, landmarks)
    print(f"Landmarks saved to {output_landmarks_path}")


def crop_video(input_video, output_video, frames_per_second=25, target_size=512):
    # def process_video_and_crop_frames(input_video, output_video, frames_per_second=25, target_size=512):
    output_dir = os.path.dirname(input_video)
    name = input_video.split('/')[-1].split('.')[0].replace('_ori', '').replace('_resized', '')
    landmarks_file = os.path.join(output_dir, f"{name}.npy")
    frame_folder = os.path.join(output_dir, f'{name}_frame')
    os.makedirs(frame_folder, exist_ok=True)

    # Initialize the face alignment pipeline
    face_ali = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device, flip_input=False)
    # Load the video
    video_capture = cv2.VideoCapture(input_video)
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # Calculate the interval to pick frames
    interval = max(1, fps // frames_per_second)

    frame_idx = 0
    landmarks_list = []
    with tqdm(total=total_frames, desc="Processing video frames", unit="frame") as pbar:
        while frame_idx < total_frames:
            # Set the video position to the current frame index
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            is_success, frame = video_capture.read()

            if not is_success:
                print(f"Error reading frame {frame_idx}")
                frame_idx += interval
                continue

            # Detect landmarks
            landmarks = face_ali.get_landmarks(frame)

            if landmarks is None or len(landmarks) == 0:
                print(f"No faces detected in frame {frame_idx}")
                frame_idx += interval
                continue

            # Get the 30th landmark (index 29) 获取第一张脸（索引为0）的第30个特征点（鼻子？）（索引为29）
            landmark_30 = landmarks[0][29]
            landmarks_list.append(landmark_30)

            # Calculate the crop region
            x, y = int(landmark_30[0]), int(landmark_30[1])
            half_size = target_size // 2
            x1, y1 = max(0, x - half_size), max(0, y - half_size)
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
            output_image_path = os.path.join(frame_folder, f'cropped_frame_{frame_idx}.jpg')
            cv2.imwrite(output_image_path, cropped_frame)

            frame_idx += interval
            # 更新进度条
            pbar.update(interval)
    # Save landmarks to a file
    save_landmarks(np.array(landmarks_list), landmarks_file)
    video_capture.release()

    # Extract audio from the original video
    video = VideoFileClip(input_video)
    audio = video.audio
    audio.write_audiofile(os.path.join(output_dir, 'audio.wav'))

    # Get a list of all image files in the folder
    image_files = os.listdir(frame_folder)
    # Sort the list in the order(顺序) you want
    image_files.sort(key=sort_key)
    # Create full paths for each image file
    image_paths = [os.path.join(frame_folder, filename) for filename in image_files]

    # Create a video from the image sequence
    video_without_audio = ImageSequenceClip(image_paths, fps=25)
    # video_without_audio.write_videofile(os.path.join(output_dir, f"{name}_no_audio.mp4"))
    # Add audio to the video
    video_with_audio = video_without_audio.set_audio(audio)
    # video_with_audio.write_videofile(os.path.join(output_folder, f"{name}.mp4"))
    video_with_audio.write_videofile(os.path.join(output_dir, f"{name}.mp4"), codec='libx264')


def run(input_video, output_video, is_resized=True):
    # Process video and crop frames
    target_fps = 25
    crop_size = 720
    target_size = 512
    if is_resized:
        input_video = resize_video(input_video, target_fps, crop_size, target_size)
    crop_video(input_video, output_video)


if __name__ == '__main__':
    in_default_path = '/home/zxd/code/Vision/GeneFacePlusPlus/data/raw/videos/yu/yu_ori.mp4'
    out_default_path = '/home/zxd/code/Vision/GeneFacePlusPlus/output/yu/yu_out.mp4'
    resized = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=in_default_path)  # 输入的视频路径
    parser.add_argument('--output_path', type=str, default=out_default_path)  # 输出的视频路径
    args = parser.parse_args()
    run(args.input_path, args.output_path, is_resized=resized)
    # data = np.load('/home/zxd/code/Vision/GeneFacePlusPlus/data/raw/.../save_landmarks.npy')
    # print(data.shape)


