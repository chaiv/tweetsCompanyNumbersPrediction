'''
Created on 09.12.2023

@author: vital
'''
from topicmodelling.TopicExtractor import Top2VecTopicExtractor
from itertools import chain
import re
from nlpvectors.AbstractTokenizer import AbstractTokenizer


class LLMTopicsCompare(object):



    def __init__(self, topicExtractor: Top2VecTopicExtractor,tokenizer: AbstractTokenizer,llmtopicsDf, tweetIdColumnName = "tweet_id"):
        self.topicExtractor = topicExtractor
        self.llmtopicsDf = llmtopicsDf
        self.tweetIdColumnName = tweetIdColumnName
        self.tokenizer = tokenizer
        
    def calculatePercentageOfTrue(self,allSimilarities: list[list[bool]]):
        allSimilaritiesFlattened = list(chain.from_iterable(allSimilarities))
        true_count = sum( allSimilaritiesFlattened )
        total_count = len(allSimilaritiesFlattened )
        percentageOfTrue    = (true_count / total_count) if total_count > 0 else 0        
        return percentageOfTrue   
    
    
    def getTopicWords(self,llmTopicsStr):
        #splitBySemicolonAndBlankAndRemoveEmptyStringsRegex = r'[; ]+'
        #llmTopics = re.split(splitBySemicolonAndBlankAndRemoveEmptyStringsRegex ,llmTopicsStr)
        topicWords = []
        for compoundWord in llmTopicsStr.split(";"):
            topicWords.extend(self.tokenizer.tokenize(compoundWord))
        return topicWords
        
    
    def calculateSimilarityFlags(self, lLMTopicsColumnName,numTopicsToFind=3):
        tweetIds = [int(x) for x in self.llmtopicsDf[self.tweetIdColumnName].tolist()]
        doc_topics, _,topic_words,_ = self.topicExtractor.get_documents_topics(tweetIds,num_topics=numTopicsToFind)
        llmTopicsLists = self.llmtopicsDf[lLMTopicsColumnName].tolist()
        allSimilarities = []
        for i in range(0,len(tweetIds)):
            if(numTopicsToFind == 1):
                tweetTopics = [doc_topics[i]]
            else: 
                tweetTopics = doc_topics[i]
            llmTopics = self.getTopicWords(llmTopicsLists[i])
            similarityFlags = []
            for llmTopic in llmTopics: 
                _,_,_,topicIdsSimilarToLLMTopics = self.topicExtractor.searchTopics([llmTopic], numTopicsToFind)
                topicFoundFlag = False
                for topicId in topicIdsSimilarToLLMTopics:
                    if(topicId in tweetTopics):
                        topicFoundFlag = True
                        break
                similarityFlags.append(topicFoundFlag)      
            allSimilarities.append(similarityFlags)
        return  allSimilarities   
        
    def calculateSimilarityScore(self, lLMTopicsColumnName,numTopicsToFind=3):
        allSimilarities = self.calculateSimilarityFlags(lLMTopicsColumnName,numTopicsToFind)
        return self.calculatePercentageOfTrue(allSimilarities)
                
