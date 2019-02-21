import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adadelta

device = torch.device('cuda' if torch.cuda.is_avaliable() else 'cpu')

from utils.dataloader import DataLoader


def data_gen(data_loader: DataLoader, size: int = 32, loops: int = -1, device=None):
    for x, xi, y, yi in dataloader.gen_batch(size, loops):
        x = torch.from_numpy(x).to(torch.long).to(device)
        xi = torch.from_numpy(xi).to(torch.long).to(device)
        y = torch.from_numpy(y).to(torch.long).to(device)
        yi = torch.from_numpy(yi).to(torch.long).to(device)
        yield x, xi, y, yi


class TransformRnnNet(nn.Module):
    def __init__(self, pinyin_size, word_size):
        super().__init__()

        self.embedding = nn.Embedding(pinyin_size, 256)
        self.encoding_seq = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU()
        )
        self.gru1 = nn.GRU(256, 256, num_layers=2, batch_first=True, bidirectional=True)
        self.decoding_seq = nn.Conv1d(512, word_size, 1)

    def forward(self, *input):
        x = input[0]
        x = self.embedding(x)
        x = self.encoding_seq(x.transpose(1, 2))
        x, hidden = self.gru1(x.transpose(1, 2))
        x = self.decoding_seq(x.transpose(1, 2))
        x = F.log_softmax(x, dim=1)
        return x


dataloader = DataLoader('./data/zh.tsv')
ts_model = TransformRnnNet(dataloader.pinyin_numbers, dataloader.char_numbers)
from tqdm import tqdm

def train(model: TransformRnnNet, data_loader: DataLoader, loops=3000):
    model = model.train().to(device)
    optimizer = Adadelta(model.parameters(), lr=1.0)

    for x, xi, y, yi in data_gen(data_loader, loops=loops, size=32, device=device):
        optimizer.zero_grad()

        out = model(x)
        ctc_loss=F.ctc_loss(out,y,xi,yi)
        ctc_loss.backward()

        optimizer.step()
