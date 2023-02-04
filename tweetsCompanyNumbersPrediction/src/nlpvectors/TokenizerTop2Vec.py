'''
Created on 01.02.2023

@author: vital
'''
from top2vec.Top2Vec import default_tokenizer
import json

MAX_LEN = 280 #Tweet max length 

class TokenizerTop2Vec(object):
    '''
    classdocs
    '''
    def __init__(self, dictionaryPath):
        with open(dictionaryPath) as json_file:
            dictionary = json.load(json_file)
        self.word_indexes = dictionary
    
    def getVocabularyLength(self):
        return len(self.word_indexes)

    def getPADTokenID(self):
        return self.getVocabularyLength()+1   
    
    def encode(self,text):
        tokens = default_tokenizer(text)
        return [self.word_indexes[token] for token in tokens]
        