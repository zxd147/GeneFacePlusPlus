import numpy as np


def write_read_npy(npy_path, landmarks=None):
    print(f"npy_path: {npy_path}")
    # 保存 landmarks 到 .npy 文件
    if landmarks:
        print(f"保存的数据为：{landmarks}")
        np.save(npy_path, np.array(landmarks))
    # 读取 .npy 文件中的数组
    data = np.load(npy_path)
    # 输出读取到的数据
    print(f"读取到的数据为：{data}")
    print(f"数据的形状为：{data.shape}")


# 设置 .npy 文件的路径
input_npy_path = '/home/zxd/code/Vision/GeneFacePlusPlus/data/raw/videos/gu.npy'
input_landmarks = []
# 调用 write_read_npy 函数
write_read_npy(input_npy_path, input_landmarks)

