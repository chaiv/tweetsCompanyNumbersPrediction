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
    def __init__(self, dataframe,splits,splitIndexes, tokenizer:AbstractTokenizer,textEncoder:AbstractEncoder, tweetIdColumn = "tweet_id", textColumnName = "body" , classColumnName = "class", precache=True):
        self.textColumnName = textColumnName
        self.classColumnName = classColumnName
        self.textEncoder = textEncoder
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.splits = splits
        self.tweetIdColumn = tweetIdColumn
        self.splitIndexes = splitIndexes
        # Pre-build index for fast tweet_id lookups instead of repeated isin()
        self._id_to_rows = {}
        for row_idx, tweet_id in enumerate(dataframe[tweetIdColumn]):
            self._id_to_rows.setdefault(tweet_id, []).append(row_idx)
        
        # Pre-cache all feature vectors and labels to avoid repeated tokenization/encoding
        self._cache = None
        if precache:
            self._cache = []
            for idx in range(len(splitIndexes)):
                tweetGroup = self._buildTweetGroup(idx)
                x = torch.tensor(tweetGroup.getFeatureVector(), dtype=torch.long)
                y = tweetGroup.getLabel()
                self._cache.append((x, y))

    def __len__(self):
        return len(self.splitIndexes)
    
    def _buildTweetGroup(self, idx):
        split = self.splits[self.splitIndexes[idx]]
        row_indices = []
        for tweet_id in split:
            row_indices.extend(self._id_to_rows.get(tweet_id, []))
        splitDf = self.dataframe.iloc[row_indices]
        sentenceIds = split
        sentences = splitDf[self.textColumnName].tolist()
        label = splitDf[self.classColumnName].iloc[0]
        return createTweetGroup(self.tokenizer, self.textEncoder, sentences, sentenceIds, label)
    
    def getAsTweetGroup(self,idx):
        return self._buildTweetGroup(idx)

    def __getitem__(self, idx):
        if self._cache is not None:
            return self._cache[idx]
        tweetGroup = self._buildTweetGroup(idx)
        x = torch.tensor(tweetGroup.getFeatureVector(), dtype=torch.long)
        y = tweetGroup.getLabel()
        return x, y
        