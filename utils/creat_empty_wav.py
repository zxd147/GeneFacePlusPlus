import wave
import numpy as np

# 设置参数
filename = 'output/silent_16k.wav'
duration = 10  # 音频长度，单位为秒
sample_rate = 16000  # 采样率

# 创建一个采样率为sample_rate的空白音频（全0）
n_frames = int(duration * sample_rate)
audio_data = np.zeros((n_frames, 1), dtype=np.float32)

# 打开文件并写入数据
with wave.open(filename, 'wb') as wav_file:
    wav_file.setnchannels(1)  # 单声道
    wav_file.setsampwidth(2)  # 采样宽度为2字节（16位）
    wav_file.setframerate(sample_rate)  # 采样率
    wav_file.writeframes(audio_data.tobytes())  # 写入数据