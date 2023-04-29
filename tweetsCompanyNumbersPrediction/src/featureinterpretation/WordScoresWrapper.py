'''
Created on 29.04.2023

@author: vital
'''

class WordScoresWrapper(object):

    def __init__(self, total_id_sentence_id_lists,total_token_indexes_lists, total_token_lists, total_attribution_lists):
        self.total_id_sentence_id_lists = total_id_sentence_id_lists
        self.total_token_indexes_lists = total_token_indexes_lists
        self.total_token_lists = total_token_lists
        self.total_attribution_lists = total_attribution_lists
    
    def getSentenceIds(self):
        return self.total_id_sentence_id_lists    
    
    def getTokenIndexes(self):
        return self.total_token_indexes_lists
    
    def getTokens(self):
        return self.total_token_lists
    
    def getAttributions(self):
        return self.total_attribution_lists