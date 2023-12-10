'''
Created on 09.12.2023

@author: vital
'''
from topicmodelling.TopicExtractor import TopicExtractor
from itertools import chain



class LLMTopicsCompare(object):



    def __init__(self, topicExtractor: TopicExtractor,llmtopicsDf, tweetIdColumnName = "tweet_id"):
        self.topicExtractor = topicExtractor
        self.llmtopicsDf = llmtopicsDf
        self.tweetIdColumnName = tweetIdColumnName
        
    def calculatePercentageOfTrue(self,allSimilarities: list[list[bool]]):
        allSimilaritiesFlattened = list(chain.from_iterable(allSimilarities))
        true_count = sum( allSimilaritiesFlattened )
        total_count = len(allSimilaritiesFlattened )
        percentageOfTrue    = (true_count / total_count) if total_count > 0 else 0        
        return percentageOfTrue   
        
        
    def calculateSimilarityScore(self, lLMTopicsColumnName,numTopicsToFind=3):
        tweetIds = self.llmtopicsDf[self.tweetIdColumnName].tolist()
        doc_topics, _,_,_ = self.topicExtractor.get_documents_topics(tweetIds)
        llmTopicsLists = self.llmtopicsDf[lLMTopicsColumnName].tolist()
        allSimilarities = []
        for i in range(0,len(tweetIds)):
            tweetTopics = doc_topics[i]
            llmTopics = llmTopicsLists[i].split(";")
            similarityFlags = []
            for llmTopic in llmTopics: 
                _,_,_,topicIdsSimilarToLLMTopics = self.topicExtractor.searchTopics([llmTopic], numTopicsToFind)
                topicFoundFlag = False
                for topicId in topicIdsSimilarToLLMTopics:
                    if(topicId in  tweetTopics):
                        topicFoundFlag = True
                        break
                similarityFlags.append(topicFoundFlag)      
            allSimilarities.append(similarityFlags)
        return self.calculatePercentageOfTrue(allSimilarities)
                
