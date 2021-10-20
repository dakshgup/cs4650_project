import torch
import torch.nn as nn
import numpy as np

class ClassificationModel(nn.Module): 
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, \
                 num_layers=1, bidirectional=True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.bidirectional = bidirectional

    def forward(self, x):
        x_1 = self.embed(x)
        x_2_fl = self.lstm(x_1)[1][0] 
        if self.bidirectional:
          x_2 = torch.cat([x_2_fl[0], x_2_fl[-1]])
        else:
          x_2 = x_2_fl[0]
        x_3 = self.linear(x_2)
        x_4 = self.sig(x_3)
        return x_4
    





