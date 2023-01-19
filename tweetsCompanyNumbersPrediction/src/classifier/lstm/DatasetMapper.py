'''
Created on 19.01.2023

@author: vital
'''

from torch.utils.data import Dataset

class DatasetMaper(Dataset):
    '''
    Handles batches of dataset
    '''
  
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

        