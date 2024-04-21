'''
Created on 19.11.2023

@author: vital
'''
from nlpvectors.TweetTokenizer import TweetTokenizer
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
import json

def createLookUpDictionary(tweetSentences,tokenizer = TweetTokenizer(DefaultWordFilter())):
        '''
        The key can be overwritten multiple times. That is ok for current use cases, because the relevant semantics are then still present
        '''
        lookUpDict = {}
        for sentence in tweetSentences: 
            sentenceWords = tokenizer.splitSentence(sentence)
            tokenIndexes, tokens = tokenizer.tokenizeWithIndex(sentence)
            for x in range(0,len(tokens)): 
                lookUpDict[tokens[x]]=sentenceWords[tokenIndexes[x]]
        return lookUpDict 


class TokenizedWordsLookup(object):
    '''
    For cases where for the tokenized word the original word must be found. 
    '''

    def __init__(self, dictionaryPath = None, tokenizerLookupDict = None):
        if dictionaryPath is not None: 
            with open(dictionaryPath) as json_file:
                self.tokenizerLookupDict= json.load(json_file)
        if tokenizerLookupDict is not None: 
                self.tokenizerLookupDict = tokenizerLookupDict
      
    def hasOriginalWord(self,token):
        return token in self.tokenizerLookupDict
                
    def getOriginalWord(self,token):
        return self.tokenizerLookupDict[token]
    
    def getOriginalWords(self,tokens):
        originalWords = []
        for token in tokens: 
            if self.hasOriginalWord(token):
                originalWords.append(self.getOriginalWord(token))
            else: 
                originalWords.append(None)
            

    

    

    