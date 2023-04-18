'''
Created on 05.03.2023

@author: vital
'''
from nlpvectors.AbstractTokenizer import AbstractTokenizer
from tweetpreprocess.wordfiltering.AbstractTextFilter import AbstractTextFilter

class TweetTokenizer(AbstractTokenizer):
    
    def __init__(self,wordFilter: AbstractTextFilter):
        self.wordFilter = wordFilter
        

    def tokenize(self,text):
        _, tokens  = self.tokenizeWithIndex(text)
        return tokens
    
    def tokenizeWithIndex(self,text):
        indexes = []
        tokens = []
        splits = text.split()
        for index in range(len(splits)): 
            filteredWord = self.wordFilter.filter(splits[index])
            if(len(filteredWord)>0):
                indexes.append(index) 
                tokens.append(filteredWord)
        return indexes, tokens           
        