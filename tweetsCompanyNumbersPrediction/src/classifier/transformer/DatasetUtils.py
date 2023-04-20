'''
Created on 06.03.2023

@author: vital
'''
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from nlpvectors.AbstractTokenizer import AbstractTokenizer
from nlpvectors.AbstractEncoder import AbstractEncoder
from torch.utils.data import DataLoader
from functools import partial

class Dataset(Dataset):
    def __init__(self, dataframe,tokenizer:AbstractTokenizer,textEncoder:AbstractEncoder, textColumnName = "body" , classColumnName = "class"):
        self.dataframe = dataframe
        self.textColumnName = textColumnName
        self.classColumnName = classColumnName
        self.textEncoder = textEncoder
        self.tokenizer = tokenizer

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        text = str(self.dataframe[self.textColumnName].iloc[idx])
        label = self.dataframe[self.classColumnName].iloc[idx]

        x = self.textEncoder.encodeTokens(self.tokenizer.tokenize(text))
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

def createDataloader(data,batch_size, num_workers, pad_token_idx):
    return DataLoader(
        data, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=False, 
        collate_fn=partial(generate_batch, pad_idx=pad_token_idx)
        )