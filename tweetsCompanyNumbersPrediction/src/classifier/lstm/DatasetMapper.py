'''
Created on 19.01.2023

@author: vital
'''
import torch
from torch.utils.data import Dataset

class DatasetMaper(Dataset):
    '''
    Handles batches of dataset
    '''
  
    def __init__(self, x, y):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

        