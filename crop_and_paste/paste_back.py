import json
import shutil
import subprocess
import time
import cv2
import dlib
import os
import argparse
import concurrent.futures
from utils.log_utils import logger
import numpy as np
from moviepy.editor import VideoFileClip, CompositeVideoClip, AudioFileClip


def paste_back_by_ffmpeg(ori_path, infer_path, output_path, landmarks):
    exec_start = time.time()
    duration, fps, total_frames, _, _ = get_video_info_by_ffmpeg(infer_path)
    start_x, start_y, width, height, start_frame = landmarks
    end_frame = start_frame + total_frames
    clip_start = start_frame / fps
    clip_end = clip_start + duration
    logger.info("Starting coordinates for overlay:", start_x, start_y)
    # 构建ffmpeg命令行
    command = [
        'ffmpeg',
        '-i', ori_path,  # 输入原始视频
        '-i', infer_path,  # 输入推理视频
        '-filter_complex',
        f"[0:v][1:v] overlay={start_x}:{start_y}:enable='between(n,{start_frame},{end_frame})',select='between(n,{start_frame},{end_frame})'",
        # f"[0:v][1:v] overlay={start_x}:{start_y}:enable='between(t,{clip_start},{clip_end}):1',select='between(t,{clip_start},{clip_end})'",
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-y',  # 覆盖输出文件
        output_path
    ]
    # 运行命令
    try:
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True, text=True)
        logger.info(
            f"Video overlay process completed, remove infer video {infer_path}, final video saved at {output_path}.")
        _ = os.remove(infer_path) if os.path.exists(infer_path) else None
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"An error occurred while converting the video {infer_path}: {e}")
        return infer_path
    finally:
        exec_end = time.time()
        logger.debug(f'Elapsed time: {exec_end - exec_start} seconds.')


def paste_back_by_packages(ori_path, infer_path, output_path, landmarks):
    exec_start = time.time()
    duration, fps, total_frames, width, height = get_video_info_by_cv2(infer_path)
    start_x, start_y, _, _, start_frame = landmarks
    end_frame = start_frame + total_frames
    clip_start = start_frame / fps
    clip_end = clip_start + duration

    logger.info("Starting coordinates for overlay:", start_x, start_y)
    # 读取ori_path视频和infer_path视频, 读取infer_path的音频
    ori_video_clip = VideoFileClip(ori_path)
    infer_video_clip = VideoFileClip(infer_path)
    infer_audio_clip = AudioFileClip(infer_path)

    try:
        # 创建一个复合视频
        infer_video_clip = infer_video_clip.subclip(clip_start, clip_end)
        infer_video_clip = infer_video_clip.set_position((start_x, start_y))
        output_video_clip = CompositeVideoClip([ori_video_clip, infer_video_clip], size=ori_video_clip.size)
        # 将ori视频的音频替换为infer视频的音频
        output_video_clip = output_video_clip.set_audio(infer_audio_clip)
        # 将最终视频的长度设置为音频的长度
        output_video_clip = output_video_clip.set_duration(infer_video_clip.duration)
        output_video_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
        ori_video_clip.close()
        infer_video_clip.close()
        logger.info(
            f"Video overlay process completed, remove infer video {infer_path}, final video saved at {output_path}.")
        os.remove(infer_path) if os.path.exists(infer_path) else None
        return output_path
    except Exception as e:
        logger.error(f"An error occurred while converting the video {infer_path}: {e}")
        return infer_path
    finally:
        exec_end = time.time()
        logger.debug(f'Elapsed time: {exec_end - exec_start} seconds.')


def paste_back_by_packages_with_fusion(ori_path, infer_path, output_path, coordinates):
    exec_start = time.time()
    base_name = os.path.basename(infer_path).split(".")[0].split("_")[0].split("-")[0]
    base_dir = os.path.dirname(infer_path)
    frames_dir = os.path.join(base_dir, f'{base_name}_frames')
    os.makedirs(frames_dir, exist_ok=True)
    temp_output_path = output_path.replace('.mp4', '_temp.mp4')

    duration, fps, total_frames, _, _ = get_video_info_by_cv2(infer_path)
    _, _, _, width, height = get_video_info_by_cv2(ori_path)
    start_x, start_y, _, _, start_frame = coordinates
    # end_frame = start_frame + total_frames
    # clip_start = start_frame / fps
    # clip_end = clip_start + duration
    blend_range = 50  # 融合范围
    iterations = 3  # 融合次数
    feather_size = 50  # 羽化边缘
    max_workers = os.cpu_count()

    logger.info(f"Starting coordinates for overlay: {start_x, start_y}", )
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        num_each = int(total_frames / max_workers)
        frame_numbers = {}
        for i in range(max_workers - 1):
            frame_numbers[i] = (i * num_each, (i + 1) * num_each - 1)
        frame_numbers[max_workers - 1] = ((max_workers - 1) * num_each, total_frames - 1)

        futures = [executor.submit(fusion_paste_task, frame_numbers[i], ori_path, infer_path, frames_dir,
                                   coordinates, blend_range, iterations, feather_size) for i in range(max_workers)]

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                logger.debug(f"Task completed, result: {result}")
            except Exception as exc:
                logger.error(f"Task generated exception: {exc}")

    merge_exec_end = time.time()
    merge_elapsed_time = merge_exec_end - exec_start
    logger.debug(f"The fusion step code block execution time: {merge_elapsed_time} seconds")
    try:
        # ffmpeg_cmd = f"ffmpeg -y -r {fps} -i {frames_dir}/frame_%04d.jpg -c:v libx264 -pix_fmt yuv420p {temp_output_path}"
        # subprocess.call(ffmpeg_cmd, shell=True)
        # ffmpeg_cmd = f"ffmpeg -y -i {temp_output_path} -i {infer_path} -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 {output_path}"
        # subprocess.call(ffmpeg_cmd, shell=True)
        # fourcc = cv2.VideoWriter_fourcc(*'avc1')  # 使用 avc1 编码
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
        video_writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
        video_frames = [os.path.join(frames_dir, f) for f in sorted(os.listdir(frames_dir)) if f.endswith('.jpg')]
        # 写入所有帧到视频
        logger.debug(f"Number of frames: {len(video_frames)}, output path: {temp_output_path}")
        [video_writer.write(cv2.imread(frame_path)) for frame_path in video_frames]
        video_writer.release()
        video_writer.release()
        infer_audio_clip = AudioFileClip(infer_path)
        temp_video_clip = VideoFileClip(temp_output_path)
        temp_video_clip = temp_video_clip.set_audio(infer_audio_clip)
        temp_video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        infer_audio_clip.close()
        logger.info(
            f"Video overlay process completed, remove infer video {infer_path}, final video saved at {output_path}.")
        os.remove(infer_path) if os.path.exists(infer_path) else None
        return output_path
    except Exception as e:
        logger.error(f"An error occurred while converting the video {infer_path}: {e}")
        return infer_path
    finally:
        compose_exec_end = time.time()
        compose_elapsed_time = compose_exec_end - merge_exec_end
        os.remove(temp_output_path) if os.path.exists(temp_output_path) else None
        shutil.rmtree(frames_dir) if os.path.exists(frames_dir) else None
        logger.debug(f"The execution time of the merge step code block is: {compose_elapsed_time} seconds")


def get_video_info_by_ffmpeg(video_path):
    """一次性获取视频的时长、帧率和总帧数"""
    command = [
        'ffprobe',
        '-v', 'error',  # 禁止输出不必要的信息
        '-select_streams', 'v:0',  # 选择视频流
        '-show_entries', 'stream=r_frame_rate,duration,nb_frames,width,height',  # 获取时长、帧率和总帧数
        '-of', 'default=noprint_wrappers=1:nokey=1',  # 输出格式设置
        video_path
    ]
    # 运行命令
    try:
        result = subprocess.run(command, capture_output=True, check=True, text=True)
        output = result.stdout.strip().split('\n')
        width = int(output[0])  # 获取视频宽度
        height = int(output[1])  # 获取视频高度
        fps = eval(output[2])  # 帧率可能是 "numerator/denominator" 格式，需要转换为浮动数值
        duration = float(output[3])  # 时长
        total_frames = int(output[4])  # 获取总帧数
        logger.debug(f"Video info: duration={duration}, fps={fps}, total_frames={total_frames}, width={width}, height={height}")
        return duration, fps, total_frames, width, height
    except subprocess.CalledProcessError as e:
        logger.error(f"Error occurred while getting video info: {e}")
        return None, None, None, None, None


def get_video_info_by_cv2(video_path):
    video_cap = cv2.VideoCapture(video_path)
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_cap.release()
    return duration, fps, total_frames, width, height


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
        logger.debug("Center coordinates of the first detected face:", face_x, face_y)
        return face_x, face_y
    else:
        raise ValueError("No face detected in the first frame, will exit.")


def fusion_paste_task(frame_numbers, ori_path, infer_path, frames_dir, coordinates, blend_range,
                      iterations,
                      feather_size):
    paste_start, paste_end = frame_numbers
    ori_video = cv2.VideoCapture(ori_path)
    infer_video = cv2.VideoCapture(infer_path)
    start_x, start_y, width, height, index = coordinates

    while paste_start <= paste_end:
        infer_video.set(cv2.CAP_PROP_POS_FRAMES, paste_start)
        _, infer_frame = infer_video.read()
        ori_video.set(cv2.CAP_PROP_POS_FRAMES, paste_start + index)
        _, ori_frame = ori_video.read()
        # start_x = coordinates["start_x"]  # x
        # start_y = coordinates["start_y"]  # y
        # width = coordinates["width"]  # 512
        # height = coordinates["height"]  # 512
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
                                                np.newaxis]) + resized_cropped_image * feather_mask[:, :,
                                                                                       np.newaxis])  # mask 确保中间的细节部分保持清晰，而不会因为羽化造成模糊, 因为feather_mask的1.0高斯模糊处理部分比mask的1.0小。
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
        logger.debug(f"{frame_numbers}: Saved {frame_filename}")
        paste_start += 1

    ori_video.release()
    infer_video.release()
    return f'Processed frames {frame_numbers}'


if __name__ == "__main__":
    # 解析输入参数
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
    if landmarks_coords_path.endswith(".npy"):
        landmarks_coords = np.load(landmarks_coords_path)
    else:
        with open(landmarks_coords_path, "r") as file:
            landmarks_coords_json = json.load(file)
            landmarks_coords = [value for key, value in landmarks_coords_json.items()]
    mode = "ffmpeg"
    fusion = True
    if mode == "ffmpeg":
        final_output_video = paste_back_by_ffmpeg(ori_video_path, infer_video_path, output_video_path, landmarks_coords)
    elif mode == "cv2":
        if fusion:
            final_output_video = paste_back_by_packages(ori_video_path, infer_video_path, output_video_path, landmarks_coords)
        else:
            final_output_video = paste_back_by_packages_with_fusion(ori_video_path, infer_video_path, output_video_path, landmarks_coords)
    else:
        logger.error(f"Not such mode: {mode}")
        final_output_video = None
    logger.info(f"save in {final_output_video}!")
    logger.info("paste done!")

