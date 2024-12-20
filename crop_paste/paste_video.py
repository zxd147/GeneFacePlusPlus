import os
import cv2
from tqdm import tqdm
from insightface.app import FaceAnalysis
from utils.uitls import VideoReader, extract_face_get_anchor, add_alpha_channel
import numpy as np
from utils.one_euro import OneEuroFilter

app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'], providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(256, 256))
body_video_path = r"helper/J-XXZ_3min_20240130_same_length_trani_video.mp4"
body = VideoReader(body_video_path)


def paste_video_by_alpha(idx, face_frame):
    body_frame = body.__getitem__(idx)

    face_frame = np.array(face_frame)
    src_img = face_frame
    dst_img = cv2.cvtColor(body_frame, cv2.COLOR_BGR2RGB)

    src_landmarks = app.get_landmark(src_img)
    dst_landmarks = app.get_landmark(dst_img)

    src_face, _, x1, y1, w1, h1 = extract_face_get_anchor(src_img, src_landmarks)
    dst_face, dst_mask, x2, y2, w2, h2 = extract_face_get_anchor(dst_img, dst_landmarks)

    config = {
        'freq': 0.1, 'mincutoff': 0.02, 'beta': 0.1, 'dcutoff': 0.1
    }

    f = OneEuroFilter(**config)
    x2 = int(f(x2))
    y2 = int(f(y2))

    offset = 2
    w2 = offset + w2
    h2 = offset + h2

    # 获取原始脸部的内容，然后resize到原始视频的尺寸中
    src_face_forehead = src_face[y1:y1 + h1, x1:x1 + w1]
    src_face_forehead_resize = cv2.resize(src_face_forehead, (w2, h2))  # 将推理后的图片变换为原始视频的中的大小

    src_img = dst_face[y2:y2 + h2, x2:x2 + w2]

    # 这里需要使用原始的mask减去 推理后的mask来填充部分不合规的区域? 
    dst_face_forehead = dst_face[y2:y2 + h2, x2:x2 + w2]

    _, face_mask = cv2.threshold(dst_face_forehead, 0, 255, cv2.THRESH_BINARY)  # 获取resize之后的推理后的mask

    # 将 alpha 遮罩从 0-255 范围归一化到 0.0-1.0 范围
    alpha = face_mask.astype(np.float32) / 255.0
    # 执行 alpha 融合
    out_pixel = (src_face_forehead_resize * alpha + dst_face_forehead * (1 - alpha)).astype(np.uint8)

    # 制作原始视频的mask
    exchange_img = np.zeros((dst_img.shape[0], dst_img.shape[1], dst_img.shape[2]), dtype=np.uint8)
    exchange_img[y2:y2 + h2, x2:x2 + w2] = out_pixel  # 关键一步
    forehead_mask = dst_mask[y2:y2 + h2, x2:x2 + w2]
    k = 3
    kernel = np.ones((k, k), np.uint8)
    forehead_mask = cv2.erode(forehead_mask, kernel, iterations=3)  # 原始
    k = 5
    forehead_mask = cv2.GaussianBlur(forehead_mask, (k, k), 5)

    '''====================注意参数========================'''

    foreground_content_rgba = add_alpha_channel(out_pixel, forehead_mask)
    yy1 = 0
    yy2 = foreground_content_rgba.shape[0]
    xx1 = 0
    xx2 = foreground_content_rgba.shape[1]

    # 获取要覆盖图像的alpha值，将像素值除以255，使值保持在0-1之间
    alpha_png = foreground_content_rgba[yy1:yy2, xx1:xx2, 3] / 255.0
    alpha_jpg = 1 - alpha_png

    for c in range(0, 3):
        dst_img[y2:y2 + h2, x2:x2 + w2, c] = (
                (alpha_jpg * dst_img[y2:y2 + h2, x2:x2 + w2, c]) + (alpha_png * foreground_content_rgba[yy1:yy2, xx1:xx2, c]))
    return dst_img
