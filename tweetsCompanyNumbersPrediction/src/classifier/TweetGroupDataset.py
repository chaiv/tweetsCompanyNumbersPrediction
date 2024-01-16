'''
Created on 24.04.2023

@author: vital
'''
import torch
from torch.utils.data import Dataset
from nlpvectors.AbstractTokenizer import AbstractTokenizer
from nlpvectors.AbstractEncoder import AbstractEncoder
from nlpvectors.TweetGroup import createTweetGroup


class TweetGroupDataset(Dataset):
    '''
    This dataset processes multiple tweets as a single sample
    '''
    def __init__(self, dataframe,splits,splitIndexes, tokenizer:AbstractTokenizer,textEncoder:AbstractEncoder, tweetIdColumn = "tweet_id", textColumnName = "body" , classColumnName = "class"):
        self.textColumnName = textColumnName
        self.classColumnName = classColumnName
        self.textEncoder = textEncoder
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.splits = splits
        self.tweetIdColumn = tweetIdColumn
        self.splitIndexes = splitIndexes
        

    def __len__(self):
        return len(self.splitIndexes)
    
    
    def getAsTweetGroup(self,idx):
        split = self.splits[self.splitIndexes[idx]]
        splitDf =  self.dataframe [ self.dataframe [self.tweetIdColumn].isin( split)]
        sentenceIds = split
        sentences = splitDf [self.textColumnName].tolist()
        label = splitDf[self.classColumnName].iloc[0]
        tweetGroup = createTweetGroup(self.tokenizer,self.textEncoder,sentences,sentenceIds ,label)
        print(idx)
        return tweetGroup

    def __getitem__(self, idx):
        tweetGroup = self.getAsTweetGroup(idx)
        x = tweetGroup.getFeatureVector()
        x = torch.tensor(x, dtype=torch.long)
        y = tweetGroup.getLabel()
        return x, y
        