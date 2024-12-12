import json
import subprocess
import time
import cv2
import dlib
import os
import argparse
import concurrent.futures

import numpy as np
from moviepy.editor import VideoFileClip, CompositeVideoClip, AudioFileClip


def paste_back(ori_video, infer_video, final_video_path=None, crop_coordinates=None):
    start = time.time()

    output_dir = os.path.dirname(infer_video)  # 获取视频文件的目录, output
    name = os.path.splitext(os.path.basename(ori_video))[0]  # 获取不带扩展名的视频名称, li
    # 第一帧保存路径
    first_frame_path = os.path.join(output_dir, f'{name}_first_frame.png')
    # 初始化dlib的人脸检测器
    face_detector = dlib.get_frontal_face_detector()
    # 读取视频
    video_capture = cv2.VideoCapture(ori_video)
    # 获取第一帧的人脸坐标
    ret, first_frame = video_capture.read()
    if not ret:
        print("Failed to read the first frame.")
        video_capture.release()
        exit(1)
    face_coords = get_face_coordinates(face_detector, first_frame, first_frame_path)
    center_x, center_y = face_coords
    target_size = 512
    crop_size = target_size // 2
    start_x = max(center_x - crop_size, 0)
    start_y = max(center_y - crop_size, 0)
    print("Starting coordinates for overlay:", start_x, start_y)
    final_path = overlay_inferred_video(ori_video, infer_video, start_x, start_y, output_dir, name)
    # final_path = ffmpeg_paste(ori_video, infer_video, start_x, start_y, output_dir, name)  # 14.37 seconds.
    print('Video overlay process completed.')
    end = time.time()
    print(f'Elapsed time: {end - start} seconds.')
    return final_path


def get_face_coordinates(face_detector, first_frame, first_frame_path):
    # cv读取的图片转为RGB格式
    rgb_image = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    cv2.imwrite(first_frame_path, rgb_image)
    # 使用dlib的人脸检测器检测人脸
    detections = face_detector(rgb_image)
    if len(detections) > 0:
        face = detections[0]
        # 计算人脸边界框的坐标
        face_x1 = face.left()
        face_y1 = face.top()
        face_x2 = face.right()
        face_y2 = face.bottom()
        # 计算并返回人脸中心点
        face_x = (face_x1 + face_x2) // 2
        face_y = (face_y1 + face_y2) // 2
        print("Center coordinates of the first detected face:", face_x, face_y)
        return face_x, face_y
    else:
        print("No face detected in the first frame, will exit.")
        exit(1)


def overlay_inferred_video(ori_video, infer_video, start_x, start_y, output_dir, name):
    # 读取训练完的视频和原视频
    ori_clip = VideoFileClip(ori_video)
    infer_clip = VideoFileClip(infer_video)
    # 将训练完视频的位置设定在原视频中的特定位置
    # 不匹配，稍微调整位置
    # start_x = 107
    # start_y = 306
    # start_x = start_x
    # 裁剪infer_video的下边100像素
    # infer_clip = infer_clip.crop(x1=0, y1=0, x2=infer_clip.w, y2=infer_clip.h - 105)
    infer_clip = infer_clip.crop(x1=0, y1=0, x2=infer_clip.w, y2=infer_clip.h)
    infer_clip = infer_clip.set_position((start_x, start_y))
    # 创建一个复合视频
    final_clip = CompositeVideoClip([ori_clip, infer_clip], size=ori_clip.size)
    # 由于Vscode上读取的视频没有声音！！！所以这里再次传入一下推理音频
    # 读取infer视频的音频
    infer_audio = AudioFileClip(infer_video)
    # 将ori视频的音频替换为infer视频的音频
    final_clip = final_clip.set_audio(infer_audio)
    # 将最终视频的长度设置为音频的长度
    final_clip = final_clip.subclip(0, infer_audio.duration)

    # 保存最终的视频
    final_path = f"{output_dir}/{name}_compose.mp4"
    final_clip.write_videofile(final_path)
    print(f"Final video saved.")
    return final_path


def infer_merge_video(ori_video_path, infer_video_path, crop_coordinates_path):
    name = os.path.basename(ori_video_path).split('.')[0].replace('_ori', '')
    base_dir = os.path.dirname(ori_video_path)
    output_frames_dir = os.path.splitext(infer_video_path)[0] + "_output_frames"
    os.makedirs(output_frames_dir, exist_ok=True)
    # crop_coordinates_path = os.path.join(base_dir, f'{name}.npy')
    crop_coordinates = np.load(crop_coordinates_path)
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
        futures = [executor.submit(get_seamless_clone_img, frame_numbers[i], ori_video_path, infer_video_path, output_frames_dir,
                                   crop_coordinates, blend_range, iterations, feather_size) for i in range(max_workers)]
        for future in concurrent.futures.as_completed(futures):
            # try:deyuany()yuzbiru
                result = future.result()
                print(f"任务完成,结果:{result}")
            # except Exception as exc:
            #     print(f"任务生成异常:{exc}")
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


def get_seamless_clone_img(frame_numbers, ori_video_path, infer_video_path, output_frames_dir, crop_coordinates, blend_range, iterations, feather_size):
    start, end = frame_numbers
    ori_video = cv2.VideoCapture(ori_video_path)
    infer_video = cv2.VideoCapture(infer_video_path)

    while start <= end:
        infer_video.set(cv2.CAP_PROP_POS_FRAMES, start)
        _, infer_frame = infer_video.read()
        ori_video.set(cv2.CAP_PROP_POS_FRAMES, start)
        _, ori_frame = ori_video.read()
        center_x = crop_coordinates[0]  # x
        center_y = crop_coordinates[1]  # y
        width = crop_coordinates[2]  # 512
        height = crop_coordinates[3]  # 512
        start_x = center_x - width // 2
        start_y = center_y - width // 2
        # print(f'start_x: {start_x}, start_y: {start_y}')

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
        print(f"{frame_numbers}已保存{frame_filename}")
        start += 1

    ori_video.release()
    infer_video.release()
    return f'Processed frames {frame_numbers}'


def ffmpeg_paste(ori_video, infer_video, start_x, start_y, output_dir, name):
    # 构建ffmpeg命令行
    final_path = os.path.join(output_dir, f"{name}_compose.mp4")

    command = [
        'ffmpeg',
        '-i', ori_video,  # 输入原始视频
        '-i', infer_video,  # 输入推理视频
        '-filter_complex',
        f"[0:v][1:v] overlay={start_x}:{start_y}:enable='between(t,0,{get_video_duration(infer_video)})'",
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-y',  # 覆盖输出文件
        final_path
    ]

    # 执行ffmpeg命令
    subprocess.run(command, check=True)
    print(f"Final video saved at {final_path}.")
    return final_path


def get_video_duration(video_path):
    """获取视频的时长"""
    cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {video_path}"
    duration = subprocess.check_output(cmd, shell=True).decode().strip()
    return duration


if __name__ == "__main__":
    # 解析输入参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--person_id', type=str, default='li')
    parser.add_argument('--ori_video', type=str,
                        default='/home/zxd/code/Vision/GeneFacePlusPlus/temp/li_logo_trimmed_682x1212.mp4')
    parser.add_argument('--infer_video', type=str,
                        # default='/home/zxd/code/Vision/GeneFacePlusPlus/output/infer_output_li.mp4')
                        default='/home/zxd/code/Vision/GeneFacePlusPlus/temp/0930_li_torso_eye.mp4')
    opt = parser.parse_args()
    ori_path = opt.ori_video
    infer_path = opt.infer_video
    crop_coordinates_npy = ori_path.replace('mp4', 'npy')
    # final_output_video = paste_back(ori_path, infer_path)
    final_output_video = infer_merge_video(ori_path, infer_path, crop_coordinates_npy)
    print(f"save in {final_output_video}!")
    print("paste done!")

