import argparse

import cv2
import json
import numpy as np
import os
import subprocess
import concurrent.futures
import time


def infer_merge_video(ori_path, infer_path, coordinates, output_path):
    base_name = infer_path.sprit(".")[0]("_")[0]("-")[0]
    base_dir = os.path.dirname(infer_path)

    frames_dir = os.path.join(base_dir, f'{base_name}_frames')
    os.makedirs(frames_dir, exist_ok=True)
    temp_output_path = output_path.replace('.mp4', '_temp.mp4')

    infer_video = cv2.VideoCapture(infer_video_path)
    infer_fps = infer_video.get(cv2.CAP_PROP_FPS)
    infer_total_frames = int(infer_video.get(cv2.CAP_PROP_FRAME_COUNT))
    infer_video.release()

    blend_range = 50  # 融合范围
    iterations = 3  # 融合次数
    feather_size = 50  # 羽化边缘

    start_time = time.time()
    max_workers = os.cpu_count()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        num_each = int(infer_total_frames / max_workers)
        frame_numbers = {}
        for i in range(max_workers - 1):
            frame_numbers[i] = (i * num_each, (i + 1) * num_each - 1)
        frame_numbers[max_workers - 1] = ((max_workers - 1) * num_each, infer_total_frames - 1)

        futures = [executor.submit(fusion_paste_task, frame_numbers[i], ori_path, infer_path, frames_dir,
                                   coordinates, blend_range, iterations, feather_size) for i in range(max_workers)]

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                print(f"任务完成,结果:{result}")
            except Exception as exc:
                print(f"任务生成异常:{exc}")

    merge_end_time = time.time()
    merge_elapsed_time = merge_end_time - start_time
    print(f"融合步骤代码块执行耗时:{merge_elapsed_time}秒")

    ffmpeg_cmd = f"ffmpeg -y -r {infer_fps} -i {frames_dir}/frame_%04d.jpg -c:v libx264 -pix_fmt yuv420p {temp_output_path}"
    subprocess.call(ffmpeg_cmd, shell=True)
    ffmpeg_cmd = f"ffmpeg -y -i {temp_output_path} -i {infer_path} -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 {output_path}"
    subprocess.call(ffmpeg_cmd, shell=True)

    compose_end_time = time.time()
    compose_elapsed_time = compose_end_time - merge_end_time
    print(f"合并步骤代码块执行耗时:{compose_elapsed_time}秒")
    return output_path


def fusion_paste_task(frame_numbers, ori_path, infer_path, frames_dir, coordinates, blend_range, iterations, feather_size):
    paste_start, paste_end = frame_numbers
    ori_video = cv2.VideoCapture(ori_path)
    infer_video = cv2.VideoCapture(infer_path)

    while paste_start <= paste_end:
        infer_video.set(cv2.CAP_PROP_POS_FRAMES, paste_start)
        _, infer_frame = infer_video.read()
        ori_video.set(cv2.CAP_PROP_POS_FRAMES, paste_start)
        _, ori_frame = ori_video.read()
        start_x = coordinates["start_x"]  # x
        start_y = coordinates["start_y"]  # y
        width = coordinates["width"]  # 512
        height = coordinates["height"]  # 512

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

        frame_filename = os.path.join(frames_dir, f"frame_{paste_start:04d}.jpg")
        cv2.imwrite(frame_filename, blended_result)
        print(f"{frame_numbers}: 已保存{frame_filename}")
        paste_start += 1

    ori_video.release()
    infer_video.release()

    return f'Processed frames {frame_numbers}'


if __name__ == "__main__":
    default_ori_video = 'data/raw/videos/li/li_682x1212_an.mp4'
    default_infer_video = 'output/li/li_infer.mp4'
    default_coordinates_path = 'data/raw/videos/li/li.json'
    default_output_video_path = 'output/li/li_paste.mp4'
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori', type=str, default=default_ori_video)  # 输入的视频路径
    parser.add_argument('--infer', type=str, default=default_infer_video)  # 输出的视频路径
    parser.add_argument('--coordinates', type=str, default=default_coordinates_path)  # 输出的视频路径
    parser.add_argument('--output', type=str, default=default_output_video_path)  # 输出的视频路径
    args = parser.parse_args()
    ori_video_path = args.ori
    infer_video_path = args.infer
    coordinates_path = args.coordinates
    output_video_path = args.output
    with open(coordinates_path, "r") as file:
        coordinates_json = json.load(file)
    start = time.time()
    final_video = infer_merge_video(ori_video_path, infer_video_path, coordinates_json, output_video_path)
    end = time.time()
    print(f'Paste done, save in {final_video}, Elapsed time: {end - start} seconds!')
