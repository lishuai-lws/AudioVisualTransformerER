import csv
import os
from PIL import Image
import numpy as np
import librosa
import pandas as pd


from utils.logger import LOGGER
from config.config import shared_configs


def load_wav_csv(audio_path,csv_path):
    if not os.path.exists(audio_path):
        LOGGER.error("read_wav_csv audio_path:%s not exist.",audio_path)
        return
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
        LOGGER.info(f"makedirs:{csv_path}")
    audio_csv_path = os.path.join(csv_path, "audiodata.csv")
    field_names = ["audio_name", "audio_data"]
    with open(audio_csv_path,"w",newline='') as audio_csv:
        writer = csv.DictWriter(audio_csv,field_names)
        writer.writeheader()
    file_list = os.listdir(audio_path)

    df = pd.read_csv(audio_csv_path)
    for id, file in enumerate(file_list):
        wave_data, samplerate = librosa.load(os.path.join(audio_path,file))
        print(f"id:{id},name:{file}")
        df.append([file[:-4],wave_data])
    df.to_csv(audio_csv_path,header=False,index=True)

def load_video_csv(video_path,csv_path):
    LOGGER.info(f"[load_video_csv] video_path:{video_path}, csv_path:{csv_path}")
    if not os.path.exists(video_path):
        LOGGER.error("read_wav_csv audio_path:%s not exist.",video_path)
        return
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
        LOGGER.info(f"makedirs:{csv_path}")
    video_csv_path = os.path.join(csv_path, "videodata.csv")
    field_names = ["video_name", "video_data"]
    with open(video_csv_path,"w",newline='') as video_csv:
        writer = csv.DictWriter(video_csv,field_names)
        writer.writeheader()
    video_list = [dir[:-8] for dir in os.listdir(video_path) if os.path.isdir(dir)]
    LOGGER.info(f"video number:{len(video_list)} ")
    df = pd.read_csv(video_csv_path)
    for video_dir in video_list:
        print(f"video_dir:{video_dir}")
        video_dir_path = os.path.join(video_path, video_dir)
        image_list = os.listdir(video_dir_path)
        video = []
        for image in image_list:
            img = Image.open(os.path.join(video_dir_path, image))
            video.append(img)
        df.append([video_dir,video])
    df.to_csv(csv_path,header=False,index=True)






def cmumosei_data_process(args):
    audio_path = args.cmumosei_audio_path
    video_path = args.cmumosei_video_path
    csv_path = args.cmumosei_csv_path
    load_wav_csv(audio_path,csv_path)
    load_video_csv(video_path,csv_path)


if __name__ == "__main__":
    args = shared_configs.get_data_process_args()
    print(args.seed)
    # print(args.config)
    cmumosei_data_process(args)
