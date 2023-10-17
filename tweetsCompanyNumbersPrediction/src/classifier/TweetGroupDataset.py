'''
Created on 24.04.2023

@author: vital
'''
import torch
from torch.utils.data import Dataset
from nlpvectors.AbstractTokenizer import AbstractTokenizer
from nlpvectors.AbstractEncoder import AbstractEncoder
from nlpvectors.SentenceWrapper import createSentencesWrapper

class TweetGroupDataset(Dataset):
    '''
    This dataset processes multiple tweets as a single sample
    '''
    def __init__(self, dataframe,samples, tokenizer:AbstractTokenizer,textEncoder:AbstractEncoder, textColumnName = "body" , classColumnName = "class"):
        self.textColumnName = textColumnName
        self.classColumnName = classColumnName
        self.textEncoder = textEncoder
        self.tokenizer = tokenizer
        self.samples = samples
        self.dataframe = dataframe
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row_idx = self.samples[idx]
        sentences = self.dataframe.iloc[row_idx][self.textColumnName]
        label = self.dataframe.iloc[row_idx][self.classColumnName].iloc[0]
        sentencesWrapper = createSentencesWrapper(self.tokenizer,self.textEncoder,sentences,sentenceIds=None)
        x = sentencesWrapper.getFeatureVector()
        x = torch.tensor(x, dtype=torch.long)
        y = label
        return x, y
        