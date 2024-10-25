import argparse
import time

import cv2
import numpy as np
import os
from moviepy.editor import VideoFileClip


def add_audio(temp_video_path, infer_video, video_path):
    video_clip = VideoFileClip(temp_video_path)
    audio_clip = VideoFileClip(infer_video).audio
    video_clip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile(video_path, codec='libx264')


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


def paste_video(ori_video, infer_video):
    # Paths to your files
    base_dir = os.path.dirname(infer_video)
    name = ori_video.split('/')[-1].split('.')[0]
    landmarks_file = os.path.join(base_dir, f"{name}.npy")
    output_video_path = os.path.join(base_dir, f"{name}_compose.mp4")
    temp_video_path = os.path.join(base_dir, f"{name}_temp.mp4")
    # Load the landmarks from the npy file
    landmarks = np.load(landmarks_file)
    # Open the ori video and the infer video
    original_video = cv2.VideoCapture(ori_video)
    inference_video = cv2.VideoCapture(infer_video)

    # Get the video properties
    fps = original_video.get(cv2.CAP_PROP_FPS)
    width = int(original_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(original_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Create a VideoWriter for the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    # Process each (x, y) coordinate from the landmarks
    for i, (x, y) in enumerate(landmarks):
        # Read a frame from the original video and the head video
        ret1, original_frame = original_video.read()
        ret2, inference_frame = inference_video.read()
        # Check if we successfully read a frame from both videos
        if not ret1 or not ret2:
            print(f"not ret in frame {i}")
            break

        # Overlay the head frame onto the original frame
        result_frame = overlay_image(original_frame, inference_frame, int(x), int(y))
        # Write the result frame to the output video
        out.write(result_frame)
        print(f"Processed frame {i}")

    # Release the VideoCapture and VideoWriter objects
    original_video.release()
    inference_video.release()
    out.release()
    # Add audio from the head video to the output video
    add_audio(temp_video_path, infer_video, output_video_path)
    os.remove(temp_video_path)
    print(f"Video saved to {output_video_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_path', type=str,
                        default='/home/zxd/code/Vision/GeneFacePlusPlus/data/raw/videos/li/li_720x1440_25fps_30s.mp4')  # 输入的视频路径
    parser.add_argument('--infer_path', type=str, default='/home/zxd/code/Vision/GeneFacePlusPlus/output/infer_output_li.mp4')  # 输出的视频路径
    args = parser.parse_args()
    ori_path = args.ori_path
    infer_path = args.infer_path
    start = time.time()
    paste_video(ori_path, infer_path)
    end = time.time()
    print(f'Elapsed time: {end - start} seconds.')
