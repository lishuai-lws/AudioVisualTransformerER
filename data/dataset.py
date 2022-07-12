import pandas as pd
from torch.utils.data import Dataset
from config.config import shared_configs
import  os


class CMUMOSEIDataset(Dataset):
    def __init__(self, audio_path, video_path, ids_path):
        self.ids = pd.read_csv(ids_path).tolist()


    def __len__(self):
        return len(self.ids)
