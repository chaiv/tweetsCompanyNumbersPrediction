'''
Created on 01.02.2023

@author: vital
'''
from top2vec.Top2Vec import default_tokenizer


MAX_LEN = 280 #Tweet max length 

class DefaultTokenizer(object):
    '''
    classdocs
    '''


    def __init__(self, word_indexes):
        self.word_indexes = word_indexes
    
    def getVocabularyLength(self):
        return len(self.word_indexes)

    def getPADTokenID(self):
        return self.getVocabularyLength()+1   
    
    def encode(self,text):
        tokens = default_tokenizer(text)
        return [self.word_indexes[token] for token in tokens]
        