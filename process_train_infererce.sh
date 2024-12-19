#!/bin/bash

CUDA_VISIBLE_DEVICES=0
# 设置 PYTHONPATH 环境变量
export PYTHONPATH=./:$PYTHONPATH

# 获取第一个命令行参数作为 VIDEO_ID
VIDEO_ID=$1

# 输出开始处理的消息
echo "Start Process $VIDEO_ID"
# 执行 process.sh
bash process.sh $VIDEO_ID
echo "Process done!"

# 输出开始训练的消息
echo "Start Train $VIDEO_ID"
# 执行 train.sh
bash train.sh $VIDEO_ID
echo "Train done!"



# 执行推理代码
echo "Start Inference for $VIDEO_ID"
# 执行 inference.sh
bash inference.sh $VIDEO_ID
echo "Inference done!"


# 输出完成消息
echo "Process, Train, and Inference done!"