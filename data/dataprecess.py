import csv
import os
from PIL import Image
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm
import pickle

from utils.logger import LOGGER
from config.config import shared_configs


def load_wav_csv(audio_path, output_path):
    if not os.path.exists(audio_path):
        LOGGER.error("read_wav_csv audio_path:%s not exist.",audio_path)
        return
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        LOGGER.info(f"makedirs:{output_path}")
    name_csv_path = os.path.join(output_path, "dataname.csv")
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

def load_video_csv(video_path, output_path):
    LOGGER.info(f"[load_video_csv] video_path:{video_path}, csv_path:{output_path}")
    if not os.path.exists(video_path):
        LOGGER.error("read_wav_csv audio_path:%s not exist.",video_path)
        return
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        LOGGER.info(f"makedirs:{output_path}")
    video_csv_path = os.path.join(output_path, "videodata.csv")
    video_list = [dir for dir in os.listdir(video_path) if os.path.isdir(os.path.join(video_path,dir))]
    name_csv_path = os.path.join(output_path, "dataname.csv")
    name_dataframe = pd.read_csv(name_csv_path)
    LOGGER.info(f"video number:{len(video_list)} ")
    video_list = []
    i = 0
    for name in tqdm(name_dataframe[:5]):
        video_dir_path = os.path.join(video_path, name+"_aligned")
        image_list = os.listdir(video_dir_path)
        video = []
        for image in image_list:
            img = Image.open(os.path.join(video_dir_path, image))
            video.append(np.array(img.getdata()))
            img.close()
        video_list.append(video)
        if ++i >=3 :
            video_npy = np.array(video_list)
    # LOGGER.info(f"video lengths:{len(df_list)}")
    # df = pd.DataFrame(df_list)
    # df.to_csv(video_csv_path,header=False,index=False, mode="a")


def load_cmumosei_data(audio_path,video_path,output_path):
    LOGGER.info(f"[load_video_csv] audio_path:{audio_path} video_path:{video_path}, csv_path:{output_path}")
    if not os.path.exists(audio_path):
        LOGGER.error("read_wav_csv audio_path:%s not exist.",audio_path)
        return
    if not os.path.exists(video_path):
        LOGGER.error("read_wav_csv audio_path:%s not exist.",video_path)
        return
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        LOGGER.info(f"makedirs:{output_path}")
    output_file = os.path.join(output_path,"cmudata.pkl")
    file_list = os.listdir(audio_path)
    data_list = []
    for file in tqdm(file_list[:7]):
        data_dic = {}
        data_dic["name"] = file[:-4]
        audio, samplerate = librosa.load(os.path.join(audio_path,file))
        data_dic["audio"] = audio
        video_dir_path = os.path.join(video_path, file[:-4] + "_aligned")
        image_list = os.listdir(video_dir_path)
        video = []
        for image in image_list:
            img = Image.open(os.path.join(video_dir_path, image))
            video.append(np.array(img.getdata()))
            img.close()
        data_dic["video"] = np.array(video)
        data_list.append(data_dic)
        if len(data_list)>=3:
            with open(output_file,"ab") as of:
                pickle.dump(data_list, of)
            data_list.clear()
    with open(output_file, "ab") as of:
        pickle.dump(data_list, of)
    data_list.clear()
    with open(output_file,"rb") as of:
        data = pickle.load(of)
    for i in data:
        print(type(i["audio"]))
        print(type(i["video"]))

def load_cmumosei_ids(audio_path,output_path):
    file_list = os.listdir(audio_path)
    ids_list = []
    for file in tqdm(file_list):
        ids_list.append(file[:-4])
    df = pd.DataFrame(ids_list,columns=["ids"])
    df.to_csv(output_path,index=False)



def cmumosei_data_process(args):
    audio_path = args.cmumosei_audio_path
    video_path = args.cmumosei_video_path
    ids_path = args.cmumosei_ids_path
    # load_cmumosei_data(audio_path,video_path,csv_path)
    # load_wav_csv(audio_path,csv_path)
    # load_video_csv(video_path,csv_path)
    load_cmumosei_ids(audio_path,ids_path)


if __name__ == "__main__":
    args = shared_configs.get_data_process_args()
    print(args.seed)
    # print(args.config)
    cmumosei_data_process(args)
