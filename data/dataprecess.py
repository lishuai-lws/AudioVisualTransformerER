import csv
import os
from PIL import Image
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm


from utils.logger import LOGGER
from config.config import shared_configs


def load_wav_csv(audio_path, output_path):
    if not os.path.exists(audio_path):
        LOGGER.error("read_wav_csv audio_path:%s not exist.",audio_path)
        return
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        LOGGER.info(f"makedirs:{output_path}")
    name_csv_path = os.path.join(output_path, "audiodata.csv")
    audio_npy_path = os.path.join(output_path,"audiodata.npy")
    audio_list = []
    file_list = os.listdir(audio_path)
    name_list = []

    for file in tqdm(file_list[:5]):
        wave_data, samplerate = librosa.load(os.path.join(audio_path,file))
        print(wave_data.shape)
        name_list.append(file[:-4])
        audio_list.append(wave_data)

    LOGGER.info(f"audio lengths:{len(name_list)}")
    df = pd.DataFrame(name_list)
    df.to_csv(name_csv_path,index=False)
    print(type(audio_list[0]))
    audio_npy = np.array(audio_list,dtype=object)
    np.save(audio_npy_path,audio_npy)
    df2 = pd.read_csv(name_csv_path)
    print(df2)
    anpy = np.load(audio_npy_path, allow_pickle=True)
    print(type(anpy))
    print(anpy.shape)
    #27382

def load_video_csv(video_path,csv_path):
    LOGGER.info(f"[load_video_csv] video_path:{video_path}, csv_path:{csv_path}")
    if not os.path.exists(video_path):
        LOGGER.error("read_wav_csv audio_path:%s not exist.",video_path)
        return
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
        LOGGER.info(f"makedirs:{csv_path}")
    video_csv_path = os.path.join(csv_path, "videodata.csv")
    video_list = [dir for dir in os.listdir(video_path) if os.path.isdir(os.path.join(video_path,dir))]
    LOGGER.info(f"video number:{len(video_list)} ")
    df_list = []
    i = 0
    for video_dir in tqdm(video_list[:5]):
        video_dir_path = os.path.join(video_path, video_dir)
        image_list = os.listdir(video_dir_path)
        video = []
        for image in image_list:
            img = Image.open(os.path.join(video_dir_path, image))
            video.append(np.array(img.getdata()))
            img.close()
        df_list.append([video_dir[:-8],video])
        if ++i >=3 :
            df = pd.DataFrame(df_list)
            df_list.clear()
            df.to_csv(video_csv_path, header=False, index=False, mode="a")
    LOGGER.info(f"video lengths:{len(df_list)}")
    df = pd.DataFrame(df_list)
    df.to_csv(video_csv_path,header=False,index=False, mode="a")






def cmumosei_data_process(args):
    audio_path = args.cmumosei_audio_path
    video_path = args.cmumosei_video_path
    csv_path = args.cmumosei_csv_path
    load_wav_csv(audio_path,csv_path)
    # load_video_csv(video_path,csv_path)


if __name__ == "__main__":
    args = shared_configs.get_data_process_args()
    print(args.seed)
    # print(args.config)
    cmumosei_data_process(args)
