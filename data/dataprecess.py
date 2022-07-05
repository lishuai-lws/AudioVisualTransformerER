import os
import soundfile as sf
import librosa
import pandas as pd


from utils.logger import LOGGER
from config.config import shared_configs


def read_wav_csv(audio_path,audio_csv_path):
    if not audio_path.exists():
        LOGGER.error("read_wav_csv audio_path:%s not exist.",audio_path)
        return
    file_list = os.listdir(audio_path)
    df = pd.DataFrame()
    df.columns = ["id","audio_name","audio_data"]
    for id, file in enumerate(file_list):
        wave_data, samplerate = librosa.load(file)
        df.loc[id] = [id,file[:-4],wave_data]
    df.to_csv(audio_csv_path)


def cmumosei_data_process(args):
    audio_path = args.cmumosei_audio_path
    video_path = args.cmumosei_video_path


if __name__ == "__main__":
    args = shared_configs.parse_args()
    print(args.seed)