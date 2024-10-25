import numpy as np

# 设置 .npy 文件的路径
file_path = '/home/zxd/code/Vision/GeneFacePlusPlus/data/raw/videos/li/li.npy'

# 读取 .npy 文件中的数组
data = np.load(file_path)

# 输出读取到的数据
print("读取到的数据为：")
print(data)

# 输出数据的形状
print("\n数据的形状为：", data.shape)
