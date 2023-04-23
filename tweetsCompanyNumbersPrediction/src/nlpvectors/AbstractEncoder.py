'''
Created on 05.03.2023

@author: vital
'''

class AbstractEncoder(object):
    
    def __init__(self, params):
        pass
    
    def getMaxWordsAmount(self):
        pass

    def getVocabularyLength(self):
        pass

    def getPADTokenID(self):
        pass
    
    def getUNKTokenID(self):
        pass
    
    def getSEPTokenID(self):
        pass

    def encodeTokens(self,tokens):
        pass
    
        