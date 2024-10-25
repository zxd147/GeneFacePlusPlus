#!/bin/bash
# usage: CUDA_VISIBLE_DEVICES=0 bash train.sh <VIDEO_ID>
# please place video to data/raw/videos/${VIDEO_ID}/${VIDEO_ID}.mp4

VIDEO_ID=$1
echo "Training $VIDEO_ID"

# 运行头部模型训练
echo "Starting training for head model..."
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/${VIDEO_ID}/lm3d_radnerf_sr.yaml --exp_name=motion2video_nerf/${VIDEO_ID}_head --reset
echo "Training for head model completed."

# 运行躯干模型训练，使用头部模型的检查点
echo "Starting training for torso model using head model checkpoint..."
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/${VIDEO_ID}/lm3d_radnerf_torso_sr.yaml --exp_name=motion2video_nerf/${VIDEO_ID}_torso --hparams=head_model_dir=checkpoints/motion2video_nerf/${VIDEO_ID}_head --reset
echo "Training for torso model completed."
