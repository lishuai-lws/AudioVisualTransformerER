import csv
import os

import librosa
import pandas as pd


from utils.logger import LOGGER
from config.config import shared_configs


def read_wav_csv(audio_path,csv_path):
    if not os.path.exists(audio_path):
        LOGGER.error("read_wav_csv audio_path:%s not exist.",audio_path)
        return
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
        LOGGER.info(f"makedirs:{csv_path}")
    audio_csv_path = os.path.join(csv_path, "audiodata.csv")
    field_names = ["id", "audio_name", "audio_data"]
    with open(audio_csv_path,"w",newline='') as audio_csv:
        writer = csv.DictWriter(audio_csv,field_names)
        writer.writeheader()
    file_list = os.listdir(audio_path)
    df = pd.read_csv(audio_csv_path)
    for id, file in enumerate(file_list):
        wave_data, samplerate = librosa.load(os.path.join(audio_path,file))
        print(f"id:{id},name:{file}")
        df.loc[id] = [id,file[:-4],wave_data]
    df.to_csv(audio_csv_path,header=False,index=False)

def load_video_csv(video_path,csv_path):
    if not os.path.exists(video_path):
        LOGGER.error("read_wav_csv audio_path:%s not exist.",video_path)
        return
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
        LOGGER.info(f"makedirs:{csv_path}")
    video_csv_path = os.path.join(csv_path, "videodata.csv")
    field_names = ["id", "video_name", "video_data"]
    with open(video_csv_path,"w",newline='') as video_csv:
        writer = csv.DictWriter(video_csv,field_names)
        writer.writeheader()



def cmumosei_data_process(args):
    audio_path = args.cmumosei_audio_path
    video_path = args.cmumosei_video_path
    csv_path = args.cmumosei_csv_path
    read_wav_csv(audio_path,csv_path)


if __name__ == "__main__":
    args = shared_configs.get_data_process_args()
    print(args.seed)
    # print(args.config)
    cmumosei_data_process(args)
