'''
Created on 29.04.2023

@author: vital
'''
from nlpvectors.SentenceWrapper import SentencesWrapper

class WordScoresWrapper(object):

    def __init__(self, sentencesWrapper : SentencesWrapper, total_attributions):
        self.sentencesWrapper = sentencesWrapper
        self.total_attributions = total_attributions
        

    
    def getSentenceIds(self):
        return self.sentencesWrapper.getSentenceIds()
    
    def getTokenIndexes(self):
        return self.sentencesWrapper.getTokenIndexes()
    
    def getTokens(self):
        return self.sentencesWrapper.getTokens()
    
    def getAttributions(self):
        return self.total_attributions