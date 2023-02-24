'''
Created on 24.02.2023

@author: vital
'''

class AbstractTokenizer(object):
    '''
    classdocs
    '''


    def getVocabularyLength(self):
        pass

    def getPADTokenID(self):
        pass
    
    def tokenize(self,text):
        pass
    
    def tokenizeWithIndex(self,text):
        pass
        
    def encode(self,text):
        pass