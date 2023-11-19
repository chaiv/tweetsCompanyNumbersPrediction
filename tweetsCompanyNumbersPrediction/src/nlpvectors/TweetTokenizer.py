'''
Created on 05.03.2023

@author: vital
'''
from nlpvectors.AbstractTokenizer import AbstractTokenizer
from tweetpreprocess.wordfiltering.AbstractTextFilter import AbstractTextFilter

class TweetTokenizer(AbstractTokenizer):
    
    def __init__(self,wordFilter: AbstractTextFilter):
        self.wordFilter = wordFilter
        
    def splitSentence(self,sentence):
        return sentence.split()

    def tokenize(self,text):
        _, tokens  = self.tokenizeWithIndex(text)
        return tokens
    
    def tokenizeWithIndex(self,text):
        indexes = []
        tokens = []
        splits = self.splitSentence(text)
        for index in range(len(splits)): 
            filteredWord = self.wordFilter.filter(splits[index])
            if(len(filteredWord)>0):
                indexes.append(index) 
                tokens.append(filteredWord)
        return indexes, tokens           
        