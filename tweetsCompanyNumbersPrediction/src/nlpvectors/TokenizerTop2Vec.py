'''
Created on 01.02.2023

@author: vital
'''
from top2vec.Top2Vec import default_tokenizer
import json
from tagging.PosDepTagger import PosDepTagger

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
    
    def tokenize(self,text):
        return default_tokenizer(text)
    
    def tokenizeWithTagging(self,text,posDepTagger: PosDepTagger):
        tokenizedWithTags = []
        tokensWithTags = posDepTagger.get_tags(text)
        for index,token,pos,dep in tokensWithTags:
            if(len(default_tokenizer(token))>0):
                tokenizedWithTags.append([index,token,pos,dep])
        return tokenizedWithTags
        
    def encode(self,text):
        tokens = self.tokenize(text)
        return [self.word_indexes[token] for token in tokens]
        