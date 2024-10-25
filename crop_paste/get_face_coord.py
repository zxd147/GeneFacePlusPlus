import cv2
import dlib
import numpy as np


def get_face_coordinates(image):
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


if __name__ == "__main__":
    input_path = '/home/zxd/code/Vision/GeneFacePlusPlus/temp/li_-30s.mp4'
    face_npy_path = input_path.replace('.mp4', '.npy')
    video_capture = cv2.VideoCapture(input_path)
    # 获取自定义帧的人脸坐标
    diy = 1
    diy_number = 1
    frame = None
    center_x, center_y = None, None
    target_size = 521
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if diy_number == diy:
            print(f"第{diy_number}帧的大小", frame.shape)
            face_coords = get_face_coordinates(frame)
            if face_coords is not None:
                center_x, center_y = face_coords
                print(f"Center coordinates of the {diy_number} frame detected face:", center_x, center_y)
            else:
                print("No face detected in the first frame, will exit.")
                exit(1)
            break
        diy_number += 1
    landmarks = [center_x, center_y, target_size, target_size]
    # Save landmarks to a file
    np.save(face_npy_path, np.array(landmarks))
    print(f"Landmarks saved to {face_npy_path}")

