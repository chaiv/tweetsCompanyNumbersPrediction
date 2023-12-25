'''
Created on 25.12.2023

@author: vital
'''
from gensim.models import KeyedVectors
from nlpvectors.AbstractEncoder import AbstractEncoder
from nlpvectors.VocabularyCreator import PAD_TOKEN, UNK_TOKEN, SEP_TOKEN

class WordVectorsEncoder(AbstractEncoder):
    '''
    classdocs
    '''


    def __init__(self, word_vectors : KeyedVectors):
        self.word_vectors = word_vectors
    def getMaxWordsAmount(self):
        #Twitter currently limits each tweet to 280 characters, which usually translates to about 50 words on average
        return 80 
    
    def getVocabularyLength(self):
        return len(self.word_vectors)

    def getPADTokenID(self):
        return self.word_vectors[PAD_TOKEN]
    
    def getUNKTokenID(self):
        return self.word_vectors[UNK_TOKEN]
    
    def getSEPTokenID(self):
        return self.word_vectors[SEP_TOKEN]
    
    def encodeTokens(self,tokens):
        vectors = [self.word_vectors[token] if token in self.word_vectors.key_to_index else self.word_vectors[UNK_TOKEN] for token in tokens]
        return vectors
        