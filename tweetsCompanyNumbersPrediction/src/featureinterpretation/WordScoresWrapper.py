'''
Created on 29.04.2023

@author: vital
'''
from nlpvectors.TweetGroup import TweetGroup

class WordScoresWrapper(object):

    def __init__(self, tweetGroup : TweetGroup, total_attributions):
        self.tweetGroup = tweetGroup
        self.total_attributions = total_attributions
        

    
    def getSentenceIds(self):
        return self.tweetGroup.getSentenceIds()
    
    def getTokenIndexes(self):
        return self.tweetGroup.getTokenIndexes()
    
    def getTokens(self):
        return self.tweetGroup.getTokens()
    
    def getAttributions(self) -> list[list[float]]:
        return self.total_attributions
    
    def getAttributionsSum(self):
        