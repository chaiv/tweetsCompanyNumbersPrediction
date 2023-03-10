'''
Created on 05.03.2023

@author: vital
'''
import json
from nlpvectors.AbstractEncoder import AbstractEncoder
from nlpvectors.VocabularyCreator import PAD_TOKEN, UNK_TOKEN

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
        return self.vocabulary_ids[PAD_TOKEN] 
    
    def encodeTokens(self,tokens):
        return [self.vocabulary_ids.get(token, self.vocabulary_ids[UNK_TOKEN]) for token in tokens]
        