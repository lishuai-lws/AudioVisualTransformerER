import pandas as pd
from torch.utils.data import Dataset
from config.config import shared_configs
import  os


class CMUMOSEIDataset(Dataset):
    def __init__(self):
        args = shared_configs.get_data_process_args()
        audio_path = os.path.join(args.cmumosei_csv_path,"audiodata.csv")
        video_path = os.path.join(args.cmumosei_csv_path,"videodata.csv")
        audio_df = pd.read_csv(audio_path)
        vidoe_df = pd.read_csv(video_path)

