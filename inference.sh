#!/bin/bash
# usage: CUDA_VISIBLE_DEVICES=0 bash inference.sh <VIDEO_ID>

# 获取命令行参数人物ID为 VIDEO_ID 推理模型为MODEL_TYPE
VIDEO_ID=$1
MODEL_TYPE=$2

# 获取当前时间并格式化为 YYYYMMDD_HHMMSS 格式
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
# 使用 VIDEO_ID 和当前时间来构造输出文件名
OUT_NAME="temp/${VIDEO_ID}_${CURRENT_TIME}.mp4"

# 根据 MODEL_TYPE 选择模型
if [ "$MODEL_TYPE" == "head" ]; then
    MODEL_CKPT="--head_ckpt=checkpoints/motion2video_nerf/${VIDEO_ID}_head --torso_ckpt= "
else
    MODEL_CKPT="--head_ckpt= --torso_ckpt=checkpoints/motion2video_nerf/${VIDEO_ID}_torso"
fi

# 运行 Python 脚本并传递构造的文件名
CUDA_VISIBLE_DEVICES=0 python inference/genefacepp_infer.py ${MODEL_CKPT} --drv_aud=data/raw/val_wavs/start.wav --out_name ${OUT_NAME}

echo "输出文件已保存为：${OUT_NAME}"

