#!/bin/bash
# usage: CUDA_VISIBLE_DEVICES=0 bash inference.sh <VIDEO_ID>
export PYTHONPATH=./:$PYTHONPATH

# 获取命令行参数人物ID为 VIDEO_ID 推理模型为MODEL_TYPE, , 可选head/torso
VIDEO_ID=$1
MODEL_TYPE=$2

# 获取当前日期并格式化为 YYYYMMDD 格式
CURRENT_DATE=$(date +"%Y%m%d")
mkdir -p temp/${CURRENT_DATE}
# 获取当前时间并格式化为 HHMMSS 格式
CURRENT_TIME=$(date +"%H%M%S")
# 使用 VIDEO_ID、当前日期和当前时间来构造输出文件名
OUT_NAME="temp/${CURRENT_DATE}/${VIDEO_ID}_${CURRENT_TIME}.mp4"

# 根据 MODEL_TYPE 选择模型
if [ "$MODEL_TYPE" == "head" ]; then
    MODEL_CKPT="--head_ckpt=checkpoints/motion2video_nerf/${VIDEO_ID}_head --torso_ckpt= "
else
    MODEL_CKPT="--head_ckpt= --torso_ckpt=checkpoints/motion2video_nerf/${VIDEO_ID}_torso"
fi

# 运行 Python 脚本并传递构造的文件名
CUDA_VISIBLE_DEVICES=0 python inference/genefacepp_infer.py ${MODEL_CKPT} --drv_audio=data/raw/audios/start.wav --mouth_amp 0.4 --out_name ${OUT_NAME}

echo "输出文件已保存为：${OUT_NAME}"

