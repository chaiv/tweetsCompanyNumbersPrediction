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
    def __init__(self, dataframe,splits, tokenizer:AbstractTokenizer,textEncoder:AbstractEncoder, tweetIdColumn = "tweet_id", textColumnName = "body" , classColumnName = "class"):
        self.textColumnName = textColumnName
        self.classColumnName = classColumnName
        self.textEncoder = textEncoder
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.splits = splits
        self.tweetIdColumn = tweetIdColumn
        

    def __len__(self):
        return len(self.splits)

    def __getitem__(self, idx):
        split = self.splits[idx]
        splitDf =  self.dataframe [ self.dataframe [self.tweetIdColumn].isin( split)]
        sentences = splitDf [self.textColumnName]
        label = splitDf[self.classColumnName].iloc[0]
        sentencesWrapper = createSentencesWrapper(self.tokenizer,self.textEncoder,sentences,sentenceIds=None)
        x = sentencesWrapper.getFeatureVector()
        x = torch.tensor(x, dtype=torch.long)
        y = label
        return x, y
        