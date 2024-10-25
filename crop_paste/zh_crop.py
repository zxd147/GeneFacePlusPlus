import subprocess

import cv2
import dlib
import os
import json
import argparse


def crop_video(video_path, json_path):
    audio_path = video_path.replace('.mp4', '.wav')
    final_path = video_path.replace('.mp4', '_final.mp4')
    frames_dir = get_frame(video_path, json_path)
    extract_audio(video_path, audio_path)
    compose_video(final_path, audio_path, frames_dir)
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


def get_frame(video_path, json_path):
    # Initialize dlib's face detector
    detector = dlib.get_frontal_face_detector()

    # Create directory for frames
    file_name = video_path.split('.')[0]
    frames_dir = file_name + "_frames"
    print(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)
    print(f"Frames will be saved to: {frames_dir}")

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    print(f"Opened video file: {video_path}")

    # Read the first frame and get the face coordinates
    ret, first_frame = video_capture.read()
    if not ret:
        print("Failed to read the first frame from the video.")
        exit(1)

    face_coords = get_face_coordinates(detector, first_frame)
    if not face_coords:
        print("No face detected in the first frame. Exiting.")
        exit(1)

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
    if not json_path:
        print("字符串为空")
        crop_coordinates_path = frames_dir + '/crop_coordinates.json'
    else:
        crop_coordinates_path = json_path
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
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # 运行命令
    try:
        print(f"Starting extract audio...")
        subprocess.run(command, check=True)
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
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # 运行命令
    try:
        print(f"Starting compose videos...")
        subprocess.run(command, check=True)
        print(f"Compose videos successful. Output video saved to {final_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while composing videos: {e}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process a video, crop faces, and save coordinates.')
    parser.add_argument('--input_path', type=str, help='Path to the video file',
                        default='/home/zxd/code/Vision/GeneFacePlusPlus/temp/li_resized_test.mp4')
    parser.add_argument('--json_path', type=str, help='Path to the json file',
                        default='/home/zxd/code/Vision/GeneFacePlusPlus/temp/li_resized_test.json')
    args = parser.parse_args()
    input_path = args.input_path
    out_json_path = args.json_path
    final_video_path = crop_video(input_path, out_json_path)
    print(f"saved in {final_video_path}!")
    print("done!")

