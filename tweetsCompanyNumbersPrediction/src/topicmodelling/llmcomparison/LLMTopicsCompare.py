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
    
    
    def getTopicIdsAndEmbeddings(self,firstNTopicWords = 10):
        topic_words_lists,total_word_scores,topic_ids = self.topicExtractor.getTopicWordsScoresAndIds()
        topic_embeddings = []
        for i in range(0,len(topic_ids)): 
            topic_words_first_n = topic_words_lists[i][:firstNTopicWords]
            word_embeddings = self.encoder.encodeTokens(topic_words_first_n)
            topic_embeddings.append(np.mean(word_embeddings, axis=0))
        return topic_ids,topic_embeddings
    
    
    def calculateSimilarity(self, lLMTopicsColumnName,firstNTopicWords = 10,firstKTopics=3,meanLLMTopicEmbedding = True):
        allSimilarities = self.calculateSimilarityFlags(lLMTopicsColumnName,firstNTopicWords,firstKTopics,meanLLMTopicEmbedding)
        return self.calculatePercentageOfTrue(allSimilarities)
    
    def getMostSimilarEmbeddingIndexes(self, allEmbedings, embedding,firstNEmbeddings):
        similarities = cosine_similarity(embedding, allEmbedings)[0]
        similarity_sorted_indices = np.argsort(similarities)[::-1]
        return similarity_sorted_indices[:firstNEmbeddings]
    
    
    def calculateSimilarityForExactTopicWords(self, lLMTopicsColumnName):
        return self.calculatePercentageOfTrue(self.calculateSimilarityFlagsForExactTopicWords(lLMTopicsColumnName))
    
    def calculateSimilarityFlagsForExactTopicWords(self,lLMTopicsColumnName):
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
    
    
    def calculateSimilarityFlags(self,lLMTopicsColumnName,firstNTopicWords = 10,firstKTopics=3,meanLLMTopicEmbedding = True):
        tweetIds = [int(x) for x in self.llmtopicsDf[self.tweetIdColumnName].tolist()]
        _,_,tweet_topic_ids = self.topicExtractor.getDocumentTopicWordsTopicScoresAndTopicIds(tweetIds)
        topic_ids,topic_embeddings = self.getTopicIdsAndEmbeddings(firstNTopicWords) 
        llmTopicsLists = self.llmtopicsDf[lLMTopicsColumnName].tolist() 
        allSimilarities = []
        for i in range(0,len(tweetIds)):
            llmTopics = self.getTopicWords(llmTopicsLists[i])
            similarityFlags = []
            if(meanLLMTopicEmbedding):
                for llmTopic in llmTopics: 
                    llmTopicEmbedding =self.encoder.encodeTokens([llmTopic])
                    mostSimilarTopicIds = [topic_ids[idx] for idx in self.getMostSimilarEmbeddingIndexes(topic_embeddings,llmTopicEmbedding,firstKTopics)]
                    similarityFlags.append(tweet_topic_ids[i] in mostSimilarTopicIds)
            else:
                llmTopicsEmbedding =[np.mean(self.encoder.encodeTokens(llmTopics),axis=0)]
                mostSimilarTopicIds = [topic_ids[idx] for idx in self.getMostSimilarEmbeddingIndexes(topic_embeddings,llmTopicsEmbedding,firstKTopics)]
                similarityFlags.append(tweet_topic_ids[i] in mostSimilarTopicIds)
            allSimilarities.append(similarityFlags)
        return  allSimilarities            
                
                
        
        
        
                
