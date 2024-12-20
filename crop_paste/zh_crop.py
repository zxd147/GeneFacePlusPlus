import subprocess

import cv2
import dlib
import os
import json
import argparse


def crop_compose_video(input_path, output_path, resized=True, crop_size=960, target_size=512, target_fps=25):
    base_name = input_path.sprit(".")[0]("_")[0]("-")[0]
    base_dir = os.path.dirname(input_path)

    # 转分辨率后的视频路径
    resized_path = os.path.join(base_dir, f'{base_name}_resized.mp4')
    #  裁剪的音频的输出
    audio_path = os.path.join(base_dir, f'{base_name}_audio.wav')
    # 最终生成的视频路径
    final_path = output_path or os.path.join(base_dir, f'{base_name}.mp4')
    final_path = os.path.join(base_dir, base_name, '_crop', '.mp4') if os.path.isdir(final_path) else final_path
    # 保存人脸特征坐标的文件
    landmarks_path = os.path.join(base_dir, f"{base_name}.json")
    #  最终生成的视频帧的路径
    frames_dir = os.path.join(base_dir, f'{base_name}_frames')

    crop_video(input_path, frames_dir, landmarks_path)
    extract_audio(input_path, audio_path)
    compose_video(frames_dir, audio_path, final_path)
    return final_path


def get_face_coordinates(detector, image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = detector(rgb_image)
    if not detections:
        return None
    face = detections[0]
    face_x = (face.left() + face.right()) // 2
    face_y = (face.top() + face.bottom()) // 2
    return face_x, face_y


def crop_video(input_path, frames_dir, landmarks_path=None):
    # Create directory for frames
    os.makedirs(frames_dir, exist_ok=True)
    print(f"Frames will be saved to: {frames_dir}")
    # Initialize dlib's face detector
    detector = dlib.get_frontal_face_detector()

    # Open the video file
    video_capture = cv2.VideoCapture(input_path)
    print(f"Opened video file: {input_path}")
    # Read the first frame and get the face coordinates
    ret, first_frame = video_capture.read()
    if not ret:
        raise ValueError("Failed to read the first frame from the video.")

    face_coords = get_face_coordinates(detector, first_frame)
    if not face_coords:
        raise ValueError("No face detected in the first frame. Exiting.")

    print("Face detected in the first frame. Proceeding with the cropping process.")
    center_x, center_y = face_coords
    crop_size = 256  # Half of 512
    start_x = max(center_x - crop_size, 0)
    start_y = max(center_y - crop_size, 0)
    # Prepare to save the crop coordinates
    crop_coordinates = {
        "start_x": start_x,
        "start_y": start_y,
        "width": 512,
        "height": 512
    }

    # Save the crop coordinates in JSON format
    crop_coordinates_path = landmarks_path or os.path.join(frames_dir, 'crop_coordinates.json')

    with open(crop_coordinates_path, 'w') as file:
        json.dump(crop_coordinates, file, indent=4)

    frame_number = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        cropped_image = frame[start_y:start_y + 512, start_x:start_x + 512]
        frame_filename = f"{frames_dir}/frame_{frame_number:04d}.png"
        cv2.imwrite(frame_filename, cropped_image)
        print(f"Frame {frame_number} processed and saved.")
        frame_number += 1

    print(f"Finished processing {frame_number} frames. Crop coordinates saved to {crop_coordinates_path}")
    video_capture.release()
    return frames_dir


def extract_audio(video_path, audio_path):
    # 提取视频的音频
    command = [
        "ffmpeg",
        "-i", video_path,
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


def compose_video(final_path, audio_path, frames_dir):
    frame_pattern = f"{frames_dir}/frame_%04d.png"
    target_fps = 25
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


if __name__ == "__main__":
    default_in_path = '/home/zxd/code/Vision/GeneFacePlusPlus/data/raw/videos/yu/yu_ori.mp4'
    default_out_path = default_in_path.replace("_ori", '_crop')
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process a video, crop faces, and save coordinates.')
    parser.add_argument('--input', type=str, help='Path to the input video file', default=default_in_path)
    parser.add_argument('--output', type=str, help='Path to the output video file', default=default_out_path)
    args = parser.parse_args()
    input_video_path = args.input
    output_video_path = args.output
    final_video_path = crop_compose_video(input_video_path, output_video_path, resized=True, crop_size=960, target_size=512, target_fps=25)
    print(f"crop done, saved in {final_video_path}!")


