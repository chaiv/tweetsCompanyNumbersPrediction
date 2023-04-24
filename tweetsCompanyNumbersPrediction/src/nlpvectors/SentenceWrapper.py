'''
Created on 22.04.2023

@author: vital
'''
from nlpvectors.AbstractTokenizer import AbstractTokenizer
from nlpvectors.AbstractEncoder import AbstractEncoder




def createSentencesWrapper(tokenizer: AbstractTokenizer,textEncoder : AbstractEncoder, sentences,sentenceIds):
        totalTokenIndexes = []
        totalTokens = []
        totalEncodedTokens = []
        for sentence in sentences: 
            tokenIndexes, tokens = tokenizer.tokenizeWithIndex(sentence)
            encodedTokens = textEncoder.encodeTokens(tokens)
            encodedTokens.append(textEncoder.getSEPTokenID())
            totalTokenIndexes.append(tokenIndexes)
            totalTokens.append(tokens)
            totalEncodedTokens.append(encodedTokens)
        sentencesWrapper = SentencesWrapper(sentences,sentenceIds,totalTokenIndexes,totalTokens,totalEncodedTokens)
        return sentencesWrapper


class SentencesWrapper(object):

    def __init__(self,sentences,sentenceIds,totalTokenIndexes,totalTokens,totalEncodedTokens):
        self.sentences = sentences
        self.sentenceIds = sentenceIds
        self.totalTokenIndexes = totalTokenIndexes
        self.totalTokens = totalTokens
        self.totalEncodedTokens = totalEncodedTokens
        
    def getFeatureVector(self):
        flat_totalEncodedTokens = [item for sublist in self.totalEncodedTokens for item in sublist]
        return flat_totalEncodedTokens
        
    