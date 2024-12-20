import cv2
import dlib
import numpy as np


def detect_face_center(image):
    """
    检测人脸并返回人脸中心的坐标
    """
    # 初始化dlib的人脸检测器
    face_detector = dlib.get_frontal_face_detector()
    # cv读取的图片转为RGB格式
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 使用dlib的人脸检测器检测人脸
    detections = face_detector(rgb_image)
    face_x, face_y = None, None
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
    return face_x, face_y


def get_face_coordinates(video_path, npy_path, index=1):
    """
    提取自定义帧的人脸中心坐标并保存为.npy文件
    """
    center_x, center_y = None, None
    video_capture = cv2.VideoCapture(video_path)
    frame_number = 1  # 当前处理的帧编号

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            print("视频读取完毕")
            break

        # 根据 index 参数决定检测逻辑
        if index is None or frame_number == index:
            print(f"第{frame_number}帧的大小: {frame.shape}")
            face_coords = detect_face_center(frame)
            if face_coords is None:
                print(f"第{frame_number}帧未检测到人脸")
                if index is not None:
                    # 如果指定帧未检测到人脸则退出
                    raise ValueError("未检测到人脸，将退出程序")
            else:
                center_x, center_y = face_coords
                print(f"第{frame_number}帧检测到人脸中心坐标: ({center_x}, {center_y})")
                break
        frame_number += 1

    if center_x and center_y:
        # 定义 landmarks 并保存到文件
        landmarks = [center_x, center_y]
        np.save(face_npy_path, np.array(landmarks))
        print(f"人脸中心点坐标: {landmarks} 已保存至: {face_npy_path}")
    else:
        print("未成功检测到人脸关键点，未保存任何数据。")


if __name__ == "__main__":
    input_video_path = '/home/zxd/code/Vision/GeneFacePlusPlus/temp/1220/gu_night.mp4'
    face_npy_path = input_video_path.replace('.mp4', '_center.npy')
    # 自定义需要处理的帧号, 例如处理第1帧
    frame_index = 1
    get_face_coordinates(input_video_path, face_npy_path, frame_index)
