'''
Created on 29.01.2022

@author: vital
'''
import pandas as pd
from nlpvectors.FeatureVectorMapper import FeatureVectorMapper
from topicmodelling.TopicHeader import AbstractTopicHeaderFinder,TopicHeaderCalculator


class TopicExtractor(FeatureVectorMapper,AbstractTopicHeaderFinder):
   
    def getTopicHeaderByWord(self,searchWord):
        topic_words,word_scores,topic_scores,topic_nums = self.searchTopics([searchWord],1)
        return topic_nums[0], TopicHeaderCalculator().calculateHeader(topic_words[0],self.getWordVectorsOfWords(topic_words[0]))
    
    def getTopicHeaderByIds(self,documentIds):
        doc_topics, doc_dist, topic_words, topic_word_scores = self.get_documents_topics(documentIds)
        topicHeaders = [TopicHeaderCalculator().calculateHeader(words,self.getWordVectorsOfWords(words)) for words in topic_words]
        return doc_topics, topicHeaders
            
    def getWordIndexes(self):
        return self.topicModel.word_indexes


    def __init__(self, topicModel):
        self.topicModel = topicModel
        
    def get_documents(self):
        return self.topicModel.documents
          
    def getDocumentVectorByTweetIndex(self,index):
        return self.topicModel.model.docvecs[index]    
    
    def getDocumentVectorByTweetId(self,tweetId):
        return self.topicModel.model.docvecs[self.topicModel.doc_id2index[tweetId]]
      
    def getDocumentVectorsAsArray(self):
        return self.topicModel.model.dv.vectors
    
    def getDocumentVectorSize(self):
        return self.topicModel.model.docvecs.vector_size
    
    def getNumTopics(self): 
        return self.topicModel.get_num_topics()
    
    def get_topics(self):
        topic_words, word_scores, topic_nums = self.topicModel.get_topics()
        return  topic_words, word_scores, topic_nums
    
    def get_topic_vectors(self):
        return self.topicModel.topic_vectors
    
    def get_document_vectors(self):
        return self.topicModel.document_vectors
    
    def get_all_document_topics(self):
        return self.topicModel.doc_top
  
    def searchTopics(self,keywords, num_topics):
        topic_words,word_scores,topic_scores,topic_nums  = self.topicModel.search_topics(keywords, num_topics)
        return topic_words,word_scores,topic_scores,topic_nums 
    
    def get_documents_topics(self, doc_ids):
        doc_topics, doc_dist, topic_words, topic_word_scores = self.topicModel.get_documents_topics(doc_ids)
        return doc_topics, doc_dist, topic_words, topic_word_scores
    
    
    def search_documents_by_topic(self,topic_num, num_docs):
        documents, document_scores, document_ids = self.topicModel.search_documents_by_topic(topic_num, num_docs)
        return documents, document_scores, document_ids
    
    
    def getWordVectorsOfWords(self,words):
        return self.topicModel._words2word_vectors(words)
    
    def getWordVectorsArray(self):
        return self.topicModel.word_vectors
    
    
    
        