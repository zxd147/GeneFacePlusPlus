import cv2
import json
import numpy as np
from pydub import AudioSegment


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def get_audio_duration(file_path):
    audio = AudioSegment.from_file(file_path)
    duration = len(audio) / 1000  # 将毫秒转换为秒
    return duration


def extract_face_get_anchor(img, landmarks):
    lmk = np.round(landmarks).astype(np.int32)
    hull = cv2.convexHull(lmk)
    mask = np.zeros_like(img)
    cv2.fillConvexPoly(mask, hull, (255, 255, 255))
    face = cv2.bitwise_and(img, mask)

    # ========= add 
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    return face, mask, x, y, w, h


class VideoReader(object):
    def __init__(self, video_read_path: str) -> None:
        """
        Params:
            video_read_path(str): 视频文件路径;
        """

        self.cap = cv2.VideoCapture(video_read_path)
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_conut = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_per_second = round(self.cap.get(cv2.CAP_PROP_FPS), 2)
        self.current_frame_index = 0

    def __len__(self):
        return self.frame_conut

    @property
    def shape(self):
        return (self.frame_height, self.frame_width)

    @property
    def fps(self):
        return self.frame_per_second

    def __getitem__(self, idx):
        if idx == self.current_frame_index:
            still_reading, frame = self.cap.read()
            self.current_frame_index += 1
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            still_reading, frame = self.cap.read()
            self.current_frame_index = idx + 1

        if not still_reading:
            raise ValueError(f"Video Read Error, at frame index: {idx}")

        return frame

    def close(self):
        self.cap.release()


def add_alpha_channel(img, mask):
    h, w, _ = img.shape
    img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img_rgba[:, :, 3] = mask[:, :, 0]
    return img_rgba
