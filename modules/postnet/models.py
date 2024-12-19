import torch
import torch.nn as nn

"""
这是一个PitchContourCNNPostNet的例子
"""


class PitchContourCNNPostNet(nn.Module):

    def __init__(self, in_out_dim, pitch_dim):
        """
        初始化网络的层

        :param in_out_dim: 输入和输出维度
        :param pitch_dim: 音高维度
        """
        super(PitchContourCNNPostNet, self).__init__()

        # 假设这是一个简单的卷积神经网络（CNN）
        # 这里初始化一些层，例如卷积层、全连接层等
        self.conv1 = nn.Conv1d(in_out_dim, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64 * pitch_dim, 68 * 3)  # 输出的尺寸

        # 定义一个可能的激活函数（ReLU）
        self.relu = nn.ReLU()

    def to(self, device):
        """
        将模型移到指定的设备
        :param device: 设备（CPU或GPU）
        """
        super(PitchContourCNNPostNet, self).to(device)

    def forward(self, raw_pred_lm3d, pitch):
        """
        前向传播的计算，定义网络的执行逻辑

        :param raw_pred_lm3d: 初始预测（例如，3D人脸的某些特征）
        :param pitch: 音高特征
        :return: 处理后的输出
        """
        # 处理 raw_pred_lm3d
        x = raw_pred_lm3d

        # 添加卷积层
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        # 将输出拉平，准备进入全连接层
        x = x.view(x.size(0), -1)

        # 处理 pitch 特征（假设 pitch 和 x 进行加法操作）
        x = x + pitch

        # 全连接层得到最终的输出
        output = self.fc(x)

        return output

    def __call__(self, raw_pred_lm3d, pitch):
        """
        使模型可以像函数一样调用，实际上调用的是 forward 方法
        :param raw_pred_lm3d: 初始预测
        :param pitch: 音高特征
        :return: 模型输出
        """
        return self.forward(raw_pred_lm3d, pitch)
