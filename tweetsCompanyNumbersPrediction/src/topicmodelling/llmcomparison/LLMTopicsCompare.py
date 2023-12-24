'''
Created on 09.12.2023

@author: vital
'''
from topicmodelling.TopicExtractor import Top2VecTopicExtractor,\
    AbstractTopicExtractor
from itertools import chain
import re
from nlpvectors.AbstractTokenizer import AbstractTokenizer
from nlpvectors.AbstractEncoder import AbstractEncoder
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class LLMTopicsCompare(object):



    def __init__(self, topicExtractor: AbstractTopicExtractor,tokenizer: AbstractTokenizer,
                 encoder: AbstractEncoder,
                 llmtopicsDf, tweetIdColumnName = "tweet_id"):
        self.topicExtractor = topicExtractor
        self.llmtopicsDf = llmtopicsDf
        self.tweetIdColumnName = tweetIdColumnName
        self.tokenizer = tokenizer
        self.encoder = encoder
        
    def calculatePercentageOfTrue(self,allSimilarities: list[list[bool]]):
        allSimilaritiesFlattened = list(chain.from_iterable(allSimilarities))
        true_count = sum( allSimilaritiesFlattened )
        total_count = len(allSimilaritiesFlattened )
        percentageOfTrue    = (true_count / total_count) if total_count > 0 else 0        
        return percentageOfTrue   
    
    
    def getTopicWords(self,llmTopicsStr):
        topicWords = []
        for compoundWord in llmTopicsStr.split(";"):
            topicWords.extend(self.tokenizer.tokenize(compoundWord))
        return topicWords
        
    
    def calculateSimilarityFlagsForTop2Vec(self, lLMTopicsColumnName,numTopicsToFind=3):
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
    
    def calculateSimilarityFlagsForBert(self,lLMTopicsColumnName):
        tweetIds = [int(x) for x in self.llmtopicsDf[self.tweetIdColumnName].tolist()]
        topic_words,total_word_scores,topic_ids = self.topicExtractor.getDocumentTopicWordsTopicScoresAndTopicIds(tweetIds)
        llmTopicsLists = self.llmtopicsDf[lLMTopicsColumnName].tolist() 
        allSimilarities = []
        for i in range(0,len(tweetIds)):
            llmTopics = self.getTopicWords(llmTopicsLists[i])
            similarityFlags = []
            for llmTopic in llmTopics: 
                topicFoundFlag = llmTopic in topic_words[i]
                similarityFlags.append(topicFoundFlag)      
            allSimilarities.append(similarityFlags)
        return  allSimilarities 
     
     
    def calculateSimilarityScoreTop2Vec(self, lLMTopicsColumnName,numTopicsToFind=3):
        allSimilarities = self.calculateSimilarityFlagsForTop2Vec(lLMTopicsColumnName,numTopicsToFind)
        return self.calculatePercentageOfTrue(allSimilarities)
        
    def calculateSimilarityScoreBert(self, lLMTopicsColumnName):
        allSimilarities = self.calculateSimilarityFlagsForBert(lLMTopicsColumnName)
        return self.calculatePercentageOfTrue(allSimilarities)
    
    
    def getTopicIdsAndEmbeddings(self,firstNTopicWords = 10):
        topic_words_lists,total_word_scores,topic_ids = self.topicExtractor.getTopicWordsScoresAndIds()
        topic_embeddings = []
        for i in range(0,len(topic_ids)): 
            topic_words_first_n = topic_words_lists[i][:firstNTopicWords]
            word_embeddings = self.encoder.encodeTokens(topic_words_first_n)
            topic_embeddings.append(np.mean(word_embeddings, axis=0))
        return topic_ids,topic_embeddings
    
    
    def calculateSimilarity(self, lLMTopicsColumnName):
        allSimilarities = self.calculateSimilarityFlags(lLMTopicsColumnName)
        return self.calculatePercentageOfTrue(allSimilarities)
    
    def calculateSimilarityFlags(self,lLMTopicsColumnName,firstNTopicWords = 10,firstKTopics=3):
        tweetIds = [int(x) for x in self.llmtopicsDf[self.tweetIdColumnName].tolist()]
        _,_,_,tweet_topic_ids = self.topicExtractor.getDocumentTopicWordsTopicScoresAndTopicIds(self,tweetIds)
        topic_ids,topic_embeddings = self.getTopicEmbeddingsDict(firstNTopicWords) 
        llmTopicsLists = self.llmtopicsDf[lLMTopicsColumnName].tolist() 
        allSimilarities = []
        for i in range(0,len(tweetIds)):
            llmTopics = self.getTopicWords(llmTopicsLists[i])
            similarityFlags = []
            for llmTopic in llmTopics: 
                llmTopicEmbedding =self.encoder.encodeTokens([llmTopic])
                similarities = cosine_similarity(llmTopicEmbedding, topic_embeddings)[0]
                similarity_sorted_indices = np.argsort(similarities)[::-1]
                mostSimilarTopicIds = [topic_ids[idx] for idx in similarity_sorted_indices[:firstKTopics]]
                similarityFlags.append(tweet_topic_ids[i] in mostSimilarTopicIds)  
            allSimilarities.append(similarityFlags)
        return  allSimilarities            
                
                
        
        
        
                
