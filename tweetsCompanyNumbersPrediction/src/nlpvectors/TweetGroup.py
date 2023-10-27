'''
Created on 22.04.2023

@author: vital
'''
from nlpvectors.AbstractTokenizer import AbstractTokenizer
from nlpvectors.AbstractEncoder import AbstractEncoder
from itertools import chain



def createTweetGroup(tokenizer: AbstractTokenizer,textEncoder : AbstractEncoder, sentences,sentenceIds,label):
    
    if len(sentences)==0: 
        return TweetGroup()
    
    totalTokenIndexes = []
    totalTokens = []
    totalEncodedTokens = []
    for sentence in sentences: 
        tokenIndexes, tokens = tokenizer.tokenizeWithIndex(sentence)
        totalTokenIndexes.append(tokenIndexes)
        totalTokens.append(tokens)
        totalEncodedTokens.append(textEncoder.encodeTokens(tokens))
        
    encodedSeparatorToken = textEncoder.getSEPTokenID()
    # Append every list in totalEncodedTokens to each other with encodedSeparatorToken in between
    totalFeatureVector = list(chain.from_iterable([token_list + [encodedSeparatorToken] for token_list in totalEncodedTokens[:-1]])) + totalEncodedTokens[-1]

    # Find the indexes of encodedSeparatorToken in the totalFeatureVector
    separatorIndexesInFeatureVector = [i for i, token in enumerate(totalFeatureVector) if token == encodedSeparatorToken] 
            
    tweetGroup = TweetGroup(sentences,sentenceIds,totalTokenIndexes,totalTokens,totalFeatureVector,separatorIndexesInFeatureVector,label)
    return tweetGroup


class TweetGroup(object):


    def __init__(self,sentences=[],sentenceIds=[],totalTokenIndexes=[],totalTokens=[],totalFeatureVector=[],separatorIndexesInFeatureVector=[],label = None):
        self.sentences = sentences
        self.sentenceIds = sentenceIds
        self.totalTokenIndexes = totalTokenIndexes
        self.totalTokens = totalTokens
        self.totalFeatureVector = totalFeatureVector
        self.separatorIndexesInFeatureVector = separatorIndexesInFeatureVector
        self.label = label
        
    def getFeatureVector(self):
        return self.totalFeatureVector
    
    def getSeparatorIndexesInFeatureVector(self):
        return self.separatorIndexesInFeatureVector
    
    def getSentenceIds(self):
        return self.sentenceIds
    
    def getTokens(self):
        return self.totalTokens
    
    def getTokenIndexes(self):
        return self.totalTokenIndexes 
    
    def getLabel(self):
        return self.label
        
    