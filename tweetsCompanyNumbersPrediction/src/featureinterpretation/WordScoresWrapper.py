'''
Created on 29.04.2023

@author: vital
'''

class WordScoresWrapper(object):

    def __init__(self, total_sentence_ids,total_token_indexes, total_tokens, total_attributions):
        self.total_sentence_ids = total_sentence_ids
        self.total_token_indexes = total_token_indexes
        self.total_token_lists = total_tokens
        self.total_attributions = total_attributions
    
    def getSentenceIds(self):
        return self.total_sentence_ids    
    
    def getTokenIndexes(self):
        return self.total_token_indexes
    
    def getTokens(self):
        return self.total_token_lists
    
    def getAttributions(self):
        return self.total_attributions