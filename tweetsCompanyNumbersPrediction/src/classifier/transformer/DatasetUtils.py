'''
Created on 06.03.2023

@author: vital
'''
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class Dataset(Dataset):
    def __init__(self, dataframe,textEncoder, textColumnName = "body" , classColumnName = "class"):
        self.dataframe = dataframe
        self.textColumnName = textColumnName
        self.classColumnName = classColumnName
        self.textEncoder = textEncoder

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        text = str(self.dataframe[self.textColumnName].iloc[idx])
        label = self.dataframe[self.classColumnName].iloc[idx]

        x = self.textEncoder.encode(text)
        y = label

        x = torch.tensor(x, dtype=torch.long)

        return x, y


def generate_batch(data_batch, pad_idx):
    x_input, y_output = [], []
    for (x, y) in data_batch:
        x_input.append(x)
        y_output.append(y)
    x_input = pad_sequence(x_input, padding_value=pad_idx, batch_first=True)
    y_output = torch.tensor(y_output, dtype=torch.long)
    return x_input, y_output