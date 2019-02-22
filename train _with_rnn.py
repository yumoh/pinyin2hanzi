import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adadelta

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from tqdm import tqdm
from utils.dataloader import DataLoader


def data_gen(data_loader: DataLoader, size: int = 32, loops: int = -1, device=None):
    for x, xi, y, yi in data_loader.gen_batch(size, loops):
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
            nn.Sigmoid(),
            nn.Conv1d(512, 256, 1),
            nn.Sigmoid()
        )
        self.gru1 = nn.GRU(256, 256, num_layers=2, batch_first=True, bidirectional=True)
        self.decoding_seq = nn.Conv1d(512, word_size, 1)

    def forward(self, *input):
        x = input[0]  # x.shape as [batch,seq]
        x = self.embedding(x)  # x.shape as [batch,seq,word_dim]
        x = self.encoding_seq(x.transpose(1, 2))
        x, hidden = self.gru1(x.transpose(1, 2))
        x = self.decoding_seq(x.transpose(1, 2))
        x = F.log_softmax(x, dim=1)  # x.shape as [batch,word_dim,seq]
        x = x.transpose(1, 2)
        return x


def train(model: TransformRnnNet, data_loader: DataLoader, loops=3000):
    model = model.train().to(device)
    optimizer = Adadelta(model.parameters(), lr=1.0)

    bar = tqdm(data_gen(data_loader, loops=loops, size=32, device=device), total=loops)
    for loop, (x, xi, y, yi) in enumerate(bar):
        optimizer.zero_grad()

        out = model(x)
        ctc_loss = F.ctc_loss(out.transpose(0, 1), y, xi, yi)
        ctc_loss.backward()

        optimizer.step()

        if loop % 20 == 0:
            bar.set_postfix(ctc=f"{ctc_loss.item():0.4f}")


def test_train():
    print(f'use {device}')
    print(f"loading data..")
    data_loader = DataLoader('./data/zh.tsv')
    print(data_loader)
    print(f"building model...")
    ts_model = TransformRnnNet(data_loader.pinyin_numbers, data_loader.char_numbers)
    print(f"training...")
    train(ts_model, data_loader, 3000)


if __name__ == '__main__':
    test_train()
