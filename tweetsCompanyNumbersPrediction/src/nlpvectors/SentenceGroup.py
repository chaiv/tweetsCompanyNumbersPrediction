'''
Created on 22.04.2023

@author: vital
'''
from nlpvectors.VocabularyCreator import SEP_TOKEN
from nlpvectors.AbstractTokenizer import AbstractTokenizer
from nlpvectors.AbstractEncoder import AbstractEncoder

class SentencesWrapping(object):

    def __init__(self,tokenizer: AbstractTokenizer,textEncoder : AbstractEncoder):
        self.tokenizer = tokenizer
        self.textEncoder = textEncoder
    
    
    def getSentencesWrapper(self,sentencesLists,sentenceIdsLists):
        #TODO raise error if sentencesLists not entenceIdsLists len
        sentencesWrappers = []
        for i in range(len(sentencesLists)):
            sentenceIdList = sentenceIdsLists[i]
            sentenceList = sentencesLists[i]
            totalTokenIndexes = []
            totalTokens = []
            totalEncodedTokens = []
            for sentence in sentenceList: 
                sentenceWithSeparatorToken =sentence + SEP_TOKEN 
                tokenIndexes, tokens = self.tokenizer.tokenizeWithIndex(sentenceWithSeparatorToken)
                encodedTokens = self.textEncoder.encodeTokens(tokens)
                totalTokenIndexes.append(tokenIndexes)
                totalTokens.append(tokens)
                totalEncodedTokens.append(encodedTokens)
            sentencesWrapper = SentencesWrapper(sentenceList,sentenceIdList,totalTokenIndexes,totalTokens,totalEncodedTokens)
            sentencesWrappers.append(sentencesWrapper)
        return sentencesWrappers
    