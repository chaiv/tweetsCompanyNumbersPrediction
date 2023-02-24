'''
Created on 24.02.2023

@author: vital
'''

import pandas as pd
from featureinterpretation.TokenAttributionStore import TokenAttributionStore

class TweetTokenAttributionStore:
    def __init__(self,  predictor):
        self.tokenAttributionStore = TokenAttributionStore()
        self.predictor = predictor

    
    def add_from_df(self, df):
        tweet_ids = df["tweet_id"].tolist()
        sentences = df["body"].tolist()
        word_scores = self.predictor.calculateWordScores(sentences, 1)
        tweet_ids_with_word_scores = [(x,) + y for x, y in zip(tweet_ids, word_scores)]
        self.tokenAttributionStore.add_multiple_data(tweet_ids_with_word_scores)

    def to_dataframe(self):
        return self.tokenAttributionStore.to_dataframe()
        