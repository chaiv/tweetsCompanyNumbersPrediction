'''
Created on 24.02.2023

@author: vital
'''
from featureinterpretation.TokenAttributionStore import TokenAttributionStore

class ImportantWordsExtractor:
    def __init__(self,  predictor,
                   tokenAttributionStore  :  TokenAttributionStore,
                   inputTweetIdColumn = "tweet_id",
                   inputBodyIdColumn = "body"
                   ):
        self.inputTweetIdColumn = inputTweetIdColumn
        self.inputBodyIdColumn = inputBodyIdColumn
        self.tokenAttributionStore = tokenAttributionStore
        self.predictor = predictor

    
    def add_from_df(self, df):
        tweet_ids = df[self.inputTweetIdColumn].tolist()
        sentences = df[self.inputBodyIdColumn ].tolist()
        word_scores = self.predictor.calculateWordScores(sentences, 1)
        tweet_ids_with_word_scores = [(x,) + y for x, y in zip(tweet_ids, word_scores)]
        self.tokenAttributionStore.add_multiple_data(tweet_ids_with_word_scores)

    def to_dataframe(self):
        return self.tokenAttributionStore.to_dataframe()
    
    def toDfWithFirstNSortedByAttribution(self,n, ascending = True):
        df = self.to_dataframe()
        return self.toDfWithFirstNSortedByAttributionDfParam(n,df, ascending)
    
    def toDfWithFirstNSortedByAttributionDfParam(self,n,df, ascending):
        sorted_df = df.sort_values(self.tokenAttributionStore.atrributionColumnName,ascending = ascending)
        return sorted_df.head(n)
    
        