'''
Created on 24.02.2023

@author: vital
'''

class AbstractTokenizer(object):
    

    def splitSentence(self,sentence):
        pass

    def tokenize(self,sentence):
        pass
    
    def tokenizeAndGetString(self,sentence):
        pass
    
    def tokenizeWithIndex(self,sentence):
        """
        Here not only the tokens but also the positional indexes of tokens in original text are returned. 
        This is helpful for further data analysis of important words for prediction in context of original text
        """
        pass