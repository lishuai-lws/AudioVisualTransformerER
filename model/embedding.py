from torch import nn
from transformers import Wav2Vec2Tokenizer, Wav2Vec2Model



class AudioWav2Vec2(nn.Module):
    def __init__(self,modelpath):
        super().__init__()
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(modelpath,padding=True)
        self.model = Wav2Vec2Model.from_pretrained(modelpath)

    def forward(self, wavdata):
        data = self.tokenizer(wavdata)
        feature = self.model(data)
        return feature