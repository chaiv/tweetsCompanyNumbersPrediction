'''
Created on 19.11.2023

@author: vital
'''
from nlpvectors.AbstractTokenizer import AbstractTokenizer
from nlpvectors.TweetTokenizer import TweetTokenizer
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter

class TokenizedWordsLookup(object):
    '''
    For cases where for the tokenized word the original word must be found. 
    '''


    def __init__(self, tokenizer = TweetTokenizer(DefaultWordFilter())):
        self.tokenizer = tokenizer
    
    def createLookUpDictionary(self,tweetSentences):
        '''
        The key can be overwritten multiple times. That is ok for current use cases, because the relevant semantics are then still present
        '''
        lookUpDict = {}
        for sentence in tweetSentences: 
            sentenceWords = self.tokenizer.splitSentence(sentence)
            tokenIndexes, tokens = self.tokenizer.tokenizeWithIndex(sentence)
            for x in range(0,len(tokens)): 
                lookUpDict[tokens[x]]=sentenceWords[tokenIndexes[x]]
        return lookUpDict 

    