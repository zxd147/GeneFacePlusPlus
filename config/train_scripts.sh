export VIDEO_ID=0228_gu_v0
#================data Process================
# video2img 
ffmpeg -i data/raw/videos/${VIDEO_ID}.mp4 -vf fps=25,scale=w=512:h=512 -qmin 1 -q:v 1 data/raw/videos/${VIDEO_ID}_512.mp4

mv data/raw/videos/${VIDEO_ID}.mp4 data/raw/videos/${VIDEO_ID}_to_rm.mp4
mv data/raw/videos/${VIDEO_ID}_512.mp4 data/raw/videos/${VIDEO_ID}.mp4
rm data/raw/videos/${VIDEO_ID}_to_rm.mp4

# audio_process
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=./
mkdir -p data/processed/videos/${VIDEO_ID}
ffmpeg -i data/raw/videos/${VIDEO_ID}.mp4 -f wav -ar 16000 data/processed/videos/${VIDEO_ID}/aud.wav 
python data_gen/utils/process_audio/extract_hubert.py --video_id=${VIDEO_ID}
python data_gen/utils/process_audio/extract_mel_f0.py --video_id=${VIDEO_ID}

# extract images
mkdir -p data/processed/videos/${VIDEO_ID}/gt_imgs
ffmpeg -i data/raw/videos/${VIDEO_ID}.mp4 -vf fps=25,scale=w=512:h=512 -qmin 1 -q:v 1 -start_number 0 data/processed/videos/${VIDEO_ID}/gt_imgs/%08d.jpg

# 下面的脚本如果出现问题 
# > 可能是因为我们目前默认使用多进程处理segmenter（我们本地使用的是基于CPU的mediapipe）。而启用OpenGL的mediapipe暂时无法多进程加速。你可以使用--force_single_process来避免这个问题，但处理速度可能会有点慢。
# > https://github.com/yerfor/GeneFacePlusPlus/issues/57
python data_gen/utils/process_video/extract_segment_imgs.py --ds_name=nerf --vid_dir=data/raw/videos/${VIDEO_ID}.mp4 # extract image, segmap, and background

# extract lm2d_mediapipe
python data_gen/utils/process_video/extract_lm2d.py --ds_name=nerf --vid_dir=data/raw/videos/${VIDEO_ID}.mp4
# fit 3dmm 
python data_gen/utils/process_video/fit_3dmm_landmark.py --ds_name=nerf --vid_dir=data/raw/videos/${VIDEO_ID}.mp4 --reset  --debug --id_mode=global
# package
python data_gen/runs/binarizer_nerf.py --video_id=${VIDEO_ID}


#================model train================
# 还要操作一下yaml文件，不可无脑
mkdir egs/datasets/${VIDEO_ID}
cp -r egs/datasets/0202_xxz_v0/  egs/datasets/${VIDEO_ID}
# datasets/lm3d_radnerf_torso.yaml，datasets/0228_gu_v0/lm3d_radnerf.yaml 需要改,暂时先手动改
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/${VIDEO_ID}/lm3d_radnerf_sr.yaml --exp_name=motion2video_nerf/${VIDEO_ID}_head --reset
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/${VIDEO_ID}/lm3d_radnerf_torso_sr.yaml --exp_name=motion2video_nerf/${VIDEO_ID}_torso --hparams=head_model_dir=checkpoints/motion2video_nerf/${VIDEO_ID}_head --reset


#================model inference 在`test_speed.py`文件中================
