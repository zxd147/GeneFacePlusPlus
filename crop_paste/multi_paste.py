import argparse
import time

import cv2
import numpy as np
import os
from moviepy.editor import VideoFileClip
from tqdm import tqdm


def add_audio(temp_video_path, infer_video, video_path):
    video_clip = VideoFileClip(temp_video_path)
    audio_clip = VideoFileClip(infer_video).audio
    video_clip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile(video_path, codec='libx264')
    os.remove(temp_video_path)


def overlay_image(original_image, inference_image, x, y):
    h, w = inference_image.shape[:2]
    half_h, half_w = h // 2, w // 2
    # Calculate the region of interest (ROI) in the original image. 推理视频在原始视频的位置
    x1, y1 = max(0, x - half_w), max(0, y - half_h)
    x2, y2 = min(original_image.shape[1], x1 + w), min(original_image.shape[0], y1 + h)

    # Adjust new image's dimensions if ROI goes out of bounds
    new_x1, new_y1 = max(0, half_w - x), max(0, half_h - y)
    new_x2, new_y2 = new_x1 + (x2 - x1), new_y1 + (y2 - y1)
    # Overlay new image onto the original image
    original_image[y1:y2, x1:x2] = inference_image[new_y1:new_y2, new_x1:new_x2]
    return original_image


def multi_paste(ori_path, infer_path, landmarks, output_path):
    base_name = infer_path.sprit(".")[0]("_")[0]("-")[0]
    base_dir = os.path.dirname(infer_path)
    temp_video_path = os.path.join(base_dir, f"{base_name}_temp.mp4")

    # Open the ori video and the infer video
    original_video = cv2.VideoCapture(ori_path)
    inference_video = cv2.VideoCapture(infer_path)
    # Get the video properties
    fps = original_video.get(cv2.CAP_PROP_FPS)
    width = int(original_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(original_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Create a VideoWriter for the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
    out_video = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    # Process each (x, y) coordinate from the landmarks
    for i, (center_x, center_y) in enumerate(tqdm(landmarks, desc="Processing frames", unit="frame")):
        # Read a frame from the original video and the head video
        ret1, original_frame = original_video.read()
        ret2, inference_frame = inference_video.read()

        # Check if we successfully read a frame from both videos
        if not ret1 or not ret2:
            print(f"Not ret in frame {i}")
            break

        # Overlay the head frame onto the original frame
        result_frame = overlay_image(original_frame, inference_frame, int(center_x), int(center_y))
        # Write the result frame to the output video
        out_video.write(result_frame)

    # Release the VideoCapture and VideoWriter objects
    original_video.release()
    inference_video.release()
    out_video.release()
    # Add audio from the head video to the output video
    add_audio(temp_video_path, infer_path, output_path)
    print(f"Video saved to {output_path}")


if __name__ == '__main__':
    default_ori_video = 'data/raw/videos/li/li_682x1212_an.mp4'
    default_infer_video = 'output/li/li_infer.mp4'
    default_landmarks_path = 'data/raw/videos/li/li.npy'
    default_output_video_path = 'output/li/li_paste.mp4'
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori', type=str, default=default_ori_video)  # 输入的视频路径
    parser.add_argument('--infer', type=str, default=default_infer_video)  # 输出的视频路径
    parser.add_argument('--landmarks', type=str, default=default_landmarks_path)  # 输出的视频路径
    parser.add_argument('--output', type=str, default=default_output_video_path)  # 输出的视频路径
    args = parser.parse_args()
    ori_video_path = args.ori
    infer_video_path = args.infer
    landmarks_coords_path = args.landmarks
    output_video_path = args.output
    start = time.time()
    # Load the landmarks from the npy file
    landmarks_coords = np.load(landmarks_coords_path)
    multi_paste(ori_video_path, infer_video_path, landmarks_coords, output_video_path)
    end = time.time()
    print(f'Elapsed time: {end - start} seconds.')
