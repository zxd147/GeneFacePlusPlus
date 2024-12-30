import argparse
import json
import os
import time
from queue import Queue

from utils.uitls import read_json_file
from inference.genefacepp_infer import GeneFace2Infer

data_json = read_json_file("config/config.json")
base_audio_path = data_json['base_audio_path']
base_video_path = data_json['base_video_path']
inference_info = Queue()
inference_info_done = Queue()


def parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2m_ckpt", default='checkpoints/audio2motion_vae')
    parser.add_argument("--head_ckpt", default='')
    parser.add_argument("--postnet_ckpt", default='')
    parser.add_argument("--torso_ckpt", default='checkpoints/motion2video_nerf/may_torso')
    parser.add_argument("--drv_aud", default='data/raw/audios/MacronSpeech.wav')
    parser.add_argument("--drv_pose", default='nearest', help="目前仅支持源视频的pose,包括从头开始和指定区间两种,暂时不支持in-the-wild的pose")
    parser.add_argument("--blink_mode", default='period')  # none | period
    parser.add_argument("--lle_percent", default=0.2)  # nearest | random
    parser.add_argument("--temperature", default=0.2)  # nearest | random
    parser.add_argument("--mouth_amp", default=0.4)  # nearest | random
    parser.add_argument("--raymarching_end_threshold", default=0.01, help="increase it to improve fps")  # nearest | random
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--fast", action='store_true')
    parser.add_argument("--out_name", default='tmp.mp4')
    parser.add_argument("--low_memory_usage", action='store_true', help='write img to video upon generated, leads to slower fps, but use less memory')
    args = parser.parse_args()

    # 静态
    global data_json
    args.a2m_ckpt = data_json['a2m_ckpt']
    args.torso_ckpt = data_json['torso_ckpt']
    args.drv_aud = data_json['drv_aud']
    inp = {
        'a2m_ckpt': args.a2m_ckpt,
        'postnet_ckpt': args.postnet_ckpt,
        'head_ckpt': args.head_ckpt,
        'torso_ckpt': args.torso_ckpt,
        'drv_audio_name': args.drv_aud,
        'drv_pose': args.drv_pose,
        'blink_mode': args.blink_mode,
        'temperature': float(args.temperature),
        'mouth_amp': args.mouth_amp,
        'lle_percent': float(args.lle_percent),
        'debug': args.debug,
        'out_name': args.out_name,
        'raymarching_end_threshold': args.raymarching_end_threshold,
        'low_memory_usage': args.low_memory_usage,
    }
    if args.fast:
        inp['raymarching_end_threshold'] = 0.05
    # GeneFace2Infer.example_run(inp) # 可以不实例化
    return args, inp


def post_handle():
    file_path = 'config/history.txt'
    file = open(file_path, 'r', encoding='utf-8')
    history = file.readlines()
    args, inp = parameter()
    obj = GeneFace2Infer(
        audio2secc_dir=args.a2m_ckpt, postnet_dir=args.postnet_ckpt,
        head_model_dir=args.head_ckpt, torso_model_dir=args.torso_ckpt
    )

    while True:
        try:
            for line in file:
                try:
                    line = line.strip()
                    if line not in history:
                        # 开始进行推理
                        cur_info = line.replace("'", "\"")
                        try:
                            cur_json = json.loads(cur_info)
                            print('cur_json', cur_json)
                            inp['drv_aud'] = os.path.join(base_audio_path, cur_json['audio'])
                            video_output = str(int(time.time())) + ".mp4"
                            id = cur_json["id"]
                            id_path = os.path.join(base_video_path, str(id))
                            os.makedirs(id_path, exist_ok=True)
                            inp['out_name'] = os.path.join(id_path, video_output)
                            obj.infer_once(inp)  # 推理
                        except Exception as e:
                            print("【推理失败】", e)
                except Exception as e:
                    print("【通信堵塞】,暂停一秒")
                    time.sleep(1)
        except Exception as e:
            print("【读取堵塞】,暂停一秒")
            time.sleep(1)


if __name__ == "__main__":
    post_handle()
