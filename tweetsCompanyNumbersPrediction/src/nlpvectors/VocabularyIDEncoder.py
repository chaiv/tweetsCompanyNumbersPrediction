'''
Created on 05.03.2023

@author: vital
'''
import json
from nlpvectors.AbstractEncoder import AbstractEncoder

class VocabularyIDEncoder(AbstractEncoder):
   
   
    def __init__(self, dictionaryPath):
        with open(dictionaryPath) as json_file:
            dictionary = json.load(json_file)
        self.vocabulary_ids = dictionary

    def getMaxWordsAmount(self):
        #Twitter currently limits each tweet to 280 characters, which usually translates to about 50 words on average
        return 80 
    
    def getVocabularyLength(self):
        return len(self.vocabulary_ids)

    def getPADTokenID(self):
        return self.getVocabularyLength()+1   
    
    def encodeTokens(self,tokens):
        return [self.vocabulary_ids[token] for token in tokens]
        