'''
Created on 11.03.2023

@author: vital
'''
from gensim.models import KeyedVectors
from nlpvectors.AbstractEncoder import AbstractEncoder
from nlpvectors.VocabularyCreator import PAD_TOKEN, UNK_TOKEN, SEP_TOKEN

class WordVectorsIDEncoder(AbstractEncoder):
    '''
    Returns Ids of word vectors of word vectors dictionary. This is needed if word vector mbeddings are passed as weights to pytorch model and need to be accessed via id. 
    '''
    
    def __init__(self, word_vectors : KeyedVectors):
        self.word_vectors = word_vectors
    def getMaxWordsAmount(self):
        #Twitter currently limits each tweet to 280 characters, which usually translates to about 50 words on average
        return 80 
    
    def getVocabularyLength(self):
        return len(self.word_vectors)

    def getPADTokenID(self):
        return self.word_vectors.key_to_index[PAD_TOKEN]
    
    def getUNKTokenID(self):
        return self.word_vectors.key_to_index[UNK_TOKEN]
    
    def getSEPTokenID(self):
        return self.word_vectors.key_to_index[SEP_TOKEN]
    
    def encodeTokens(self,tokens):
        indexes = [self.word_vectors.key_to_index[token] if token in self.word_vectors.key_to_index else self.getUNKTokenID() for token in tokens]
        return indexes