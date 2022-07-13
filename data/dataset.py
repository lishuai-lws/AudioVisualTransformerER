import pandas as pd
from torch.utils.data import Dataset
from config.config import shared_configs
import  os
import librosa
import torchvision.transforms as transformer


class CMUMOSEIDataset(Dataset):
    def __init__(self, audio_path, video_path, ids_path):
        ids = pd.read_csv(ids_path).tolist()
        self.videoTransformer = transformer.Compose([
            transformer.
            transformer.ToTensor()
        ])
        waveDatas = []
        for id in ids:
            wave_data, samplerate = librosa.load(os.path.join(audio_path, id+".wav"))
            waveDatas.append(wave_data)

    def __len__(self):
        return len(self.ids)
