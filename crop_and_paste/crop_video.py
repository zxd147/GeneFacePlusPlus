import argparse
import os.path
import subprocess

import cv2
import dlib


def detect_face_center(face_detector, image, first_frame_face_path):
    # cv读取的图片转为RGB格式
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 使用dlib的人脸检测器检测人脸
    detections = face_detector(rgb_image)
    if len(detections) > 0:
        face = detections[0]
        # 计算人脸边界框的坐标
        face_x1 = face.left()
        face_y1 = face.top()
        face_x2 = face.right()
        face_y2 = face.bottom()
        # 确保裁剪区域在图像范围内
        face_x1 = max(face_x1, 0)
        face_y1 = max(face_y1, 0)
        face_x2 = min(face_x2, image.shape[1])
        face_y2 = min(face_y2, image.shape[0])
        # 裁剪图像
        print(f'face frame: image[{face_y1}:{face_y2}, {face_x1}:{face_x2}]')
        crop_face = image[face_y1:face_y2, face_x1:face_x2]
        if first_frame_face_path:
            # 保存裁剪后的人脸图像
            cv2.imwrite(first_frame_face_path, crop_face)
        # 计算并返回人脸中心点
        face_x = (face_x1 + face_x2) // 2
        face_y = (face_y1 + face_y2) // 2
        print("Center coordinates of the first detected face:", face_x, face_y)
        return face_x, face_y
    else:
        raise ValueError("No face detected in the first frame, will exit.")


def crop_out_direct(input_path, output_path, crop_size, index=1, crop_fps=25):
    base_name = input_path.split(".")[0].split("_")[0].split("-")[0]
    base_dir = os.path.dirname(input_path)

    # 初始化dlib的人脸检测器
    face_detector = dlib.get_frontal_face_detector()
    # 读取视频
    video_capture = cv2.VideoCapture(input_path)
    video_width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    face_coords = None
    frame_number = 1
    while video_capture.isOpened():
        if frame_number == index:
            ret, frame = video_capture.read()
            if not ret:
                video_capture.release()
                raise ValueError(f"第{frame_number}帧无法读取视频文件。")
            # 获取人脸
            face_coords = detect_face_center(face_detector, frame, None)
            center_x, center_y = face_coords
            print(f"第{frame_number}帧检测到人脸中心坐标: ({center_x}, {center_y})")
            break
        frame_number += 1

    if not face_coords:
        video_capture.release()
        raise ValueError(f"第{index}帧未检测到人脸")

    center_x, center_y = face_coords
    # 计算裁剪区域的起始点, +1为了向上取整
    start_x = center_x - (crop_size + 1) // 2
    start_y = center_y - (crop_size + 1) // 2
    # 限制裁剪区域的起始点，确保裁剪区域不会超出视频边界
    start_x = int(max(0, min(start_x, video_width - crop_size)))
    start_y = int(max(0, min(start_y, video_height - crop_size)))

    if os.path.isdir(output_path):
        output_path = os.path.join(base_dir, base_name, '_crop', '.mp4')
    # 构造 FFmpeg 命令进行裁剪
    ffmpeg_command = [
        'ffmpeg',
        '-i', input_path,
        '-vf', f"crop={crop_size}:{crop_size}:{start_x}:{start_y}, fps={crop_fps}",
        '-qmin', '1',
        '-q:v', '1',
        '-y', output_path
    ]

    ffmpeg_command_str = subprocess.list2cmdline(ffmpeg_command)
    print(f"ffmpeg_command: {ffmpeg_command_str}")
    # 执行 FFmpeg 命令
    subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True, text=True)
    # 释放视频资源
    video_capture.release()
    print(f"Video cropped and saved to {output_path}")
    return output_path


if __name__ == "__main__":
    default_video = '/home/zxd/code/Vision/GeneFacePlusPlus/data/raw/videos/xx/xx_ori.mp4'
    args = parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=default_video)
    opt = parser.parse_args()
    input_video_path = opt.input
    output_video_path = input_video_path.replace("_ori", '_crop')
    crop_out_direct(input_video_path, output_video_path, crop_size=600, index=1, crop_fps=25)
    
    
