import argparse

import cv2
import json
import numpy as np
import os
import subprocess
import concurrent.futures
import time


def task(frame_numbers, ori_video_path, infer_video_path, output_frames_dir, crop_coordinates, blend_range, iterations,
         feather_size):
    start, end = frame_numbers
    ori_video = cv2.VideoCapture(ori_video_path)
    infer_video = cv2.VideoCapture(infer_video_path)

    while start <= end:
        infer_video.set(cv2.CAP_PROP_POS_FRAMES, start)
        _, infer_frame = infer_video.read()
        ori_video.set(cv2.CAP_PROP_POS_FRAMES, start)
        _, ori_frame = ori_video.read()
        start_x = crop_coordinates["start_x"]  # x
        start_y = crop_coordinates["start_y"]  # y
        width = crop_coordinates["width"]  # 512
        height = crop_coordinates["height"]  # 512

        resized_cropped_image = cv2.resize(infer_frame, (width, height))
        # 这个遮罩的目的是实现图像的平滑融合，主要用于羽化边缘。
        feather_mask = np.zeros((height, width), dtype=np.float32)
        feather_mask[feather_size:-feather_size, feather_size:-feather_size] = 1.0  # feather_size = 50, 定义羽化边缘的宽度
        # 使用高斯模糊对feather_mask进行模糊处理，模糊核的大小为(feather_size * 2 + 1, feather_size * 2 + 1)，标准差为0（由函数自动计算）
        # 核大小 决定模糊的范围（影响图像的过渡区域大小）, 标准差 决定模糊的强度（影响模糊效果的锐利程度）。
        feather_mask = cv2.GaussianBlur(feather_mask, (feather_size * 2 + 1, feather_size * 2 + 1), 0)
        result = ori_frame.copy()
        # 图像融合操作
        result[start_y:start_y + height, start_x:start_x + width] = (
                result[start_y:start_y + height,  # 这部分计算ori_frame的原图像区域与(1 - feather_mask)的乘积，
                start_x:start_x + width] * (1 - feather_mask[:, :,  # 即在羽化遮罩为0的地方保留原图像，在羽化遮罩为1的地方减少原图像的权重。
                                                np.newaxis]) + resized_cropped_image * feather_mask[:, :, np.newaxis])
        # mask 确保中间的细节部分保持清晰，而不会因为羽化造成模糊, 因为feather_mask的1.0高斯模糊处理部分比mask的1.0小。
        mask = np.zeros(result.shape[:2], dtype=np.float32)
        mask[start_y + blend_range:start_y + height - blend_range,
        start_x + blend_range:start_x + width - blend_range] = 1.0
        # feather_size * 2 + 1 则确保高斯模糊的窗口范围足够大，以覆盖这个羽化边缘。窗口越大，模糊效果越强，边缘过渡越平滑。
        mask = cv2.GaussianBlur(mask, (blend_range * 2 + 1, blend_range * 2 + 1), 0)

        blended_result = result.copy()
        # 多次执行融合操作
        for _ in range(iterations):
            blended_result = cv2.seamlessClone(blended_result, ori_frame, (mask * 255).astype(np.uint8),
                                               (start_x + width // 2, start_y + height // 2), cv2.NORMAL_CLONE)

        frame_filename = os.path.join(output_frames_dir, f"frame_{start:04d}.jpg")
        cv2.imwrite(frame_filename, blended_result)
        print(f"{frame_numbers}: 已保存{frame_filename}")
        start += 1

    ori_video.release()
    infer_video.release()

    return f'Processed frames {frame_numbers}'


def infer_merge_video(ori_video_path, infer_video_path, crop_coordinates_path):
    output_frames_dir = os.path.splitext(infer_video_path)[0] + "_output_frames"
    os.makedirs(output_frames_dir, exist_ok=True)
    with open(crop_coordinates_path, "r") as file:
        crop_coordinates = json.load(file)

    infer_video = cv2.VideoCapture(infer_video_path)
    infer_fps = infer_video.get(cv2.CAP_PROP_FPS)
    infer_total_frames = int(infer_video.get(cv2.CAP_PROP_FRAME_COUNT))
    infer_video.release()

    blend_range = 50
    iterations = 3
    feather_size = 50

    start_time = time.time()
    max_workers = os.cpu_count()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        num_each = int(infer_total_frames / max_workers)
        frame_numbers = {}
        for i in range(max_workers - 1):
            frame_numbers[i] = (i * num_each, (i + 1) * num_each - 1)
        frame_numbers[max_workers - 1] = ((max_workers - 1) * num_each, infer_total_frames - 1)

        futures = [executor.submit(task, frame_numbers[i], ori_video_path, infer_video_path, output_frames_dir,
                                   crop_coordinates, blend_range, iterations, feather_size) for i in range(max_workers)]

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                print(f"任务完成,结果:{result}")
            except Exception as exc:
                print(f"任务生成异常:{exc}")

    merge_end_time = time.time()
    merge_elapsed_time = merge_end_time - start_time
    print(f"融合步骤代码块执行耗时:{merge_elapsed_time}秒")

    output_video = os.path.splitext(infer_video_path)[0] + "_output.mp4"
    ffmpeg_cmd = f"ffmpeg -y -r {infer_fps} -i {output_frames_dir}/frame_%04d.jpg -c:v libx264 -pix_fmt yuv420p {output_video}"
    subprocess.call(ffmpeg_cmd, shell=True)

    final_output_video = os.path.splitext(infer_video_path)[0] + "_final_output.mp4"
    ffmpeg_cmd = f"ffmpeg -y -i {output_video} -i {infer_video_path} -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 {final_output_video}"
    subprocess.call(ffmpeg_cmd, shell=True)
    compose_end_time = time.time()
    compose_elapsed_time = compose_end_time - merge_end_time
    print(f"合并步骤代码块执行耗时:{compose_elapsed_time}秒")
    return final_output_video


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process a video, crop faces, and save coordinates.')
    parser.add_argument('--ori_video', type=str, help='Path to the video file',
                        default='/home/zxd/code/Vision/GeneFacePlusPlus/temp/li_resized_test.mp4')
    parser.add_argument('--infer_video', type=str, help='Path to the json file',
                        default='/home/zxd/code/Vision/GeneFacePlusPlus/temp/li_crop.mp4')
    args = parser.parse_args()
    ori_video = args.ori_video
    infer_video = args.infer_video
    crop_coordinates_json = ori_video.replace('mp4', 'json')
    final_video = infer_merge_video(ori_video, infer_video, crop_coordinates_json)
    print(final_video)
    print("done!")


