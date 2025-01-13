'''
Created on 29.01.2022

@author: vital
'''
from nlpvectors.FeatureVectorMapper import FeatureVectorMapper
from topicmodelling.TopicHeader import AbstractTopicHeaderFinder,TopicHeaderCalculator
from nlpvectors.AbstractTokenizer import AbstractTokenizer
from tweetpreprocess.DataDirHelper import DataDirHelper
from bertopic._bertopic import BERTopic



class AbstractTopicExtractor():
    
    def getNumberOfTopics(self): 
        pass 
    
    def getTopicWordsScoresAndIds(self):
        pass
    
    def get_documents(self):
        pass
    
    def getDocumentTopicWordsTopicScoresAndTopicIds(self,doc_ids):
        pass
    
    def searchTopics(self,keywords, num_topics):
        pass

class BertTopicExtractor(AbstractTopicExtractor):
    
    def __init__(self, topicModelPath,tokenizer: AbstractTokenizer , tweetIdTopicIdMappingDf,tweetIdColumn = "tweet_id",topicIdColumn = "topic_id"):
        self.topicModelPath = topicModelPath
        self.tweetIdTopicIdMappingDf = tweetIdTopicIdMappingDf
        self.topicIdColumn = topicIdColumn
        self.topicModel = BERTopic.load(topicModelPath)
        self.tweetIdColumn = tweetIdColumn
        self.tokenizer = tokenizer
        
    def findTopics(self,keyword, num_topics):
        topic_nums, topic_scores = self.topicModel.find_topics(keyword, top_n=num_topics)
        topic_words = []
        word_scores = []
        for topic_num in topic_nums:
            topic_info = self.topicModel.get_topic(topic_num)
            words, scores = zip(*topic_info)  # Separate words and scores
            topic_words.append(list(words))
            word_scores.append(list(scores))
        return topic_words,word_scores,topic_scores,topic_nums 
        
    def get_documents(self):
        pass
    
    
        
    def getTopicWordsScoresAndIds(self):
        topics = self.topicModel.get_topics()
        total_topic_words =[]
        total_word_scores = []
        topic_ids = []
        for topic_id, topic_words in topics.items():
            topic_ids.append(topic_id)
            current_topic_words = []
            current_word_scores=[]
            for word, score in topic_words:
                current_topic_words.append(word)
                current_word_scores.append(score)
            total_topic_words.append(current_topic_words)
            total_word_scores.append(current_word_scores)
        return total_topic_words,total_word_scores,topic_ids
    
    def getNumberOfTopics(self): 
        topic_freq = self.topicModel.get_topic_freq()
        total_topics = len(topic_freq) - 1 if -1 in topic_freq.Topic.values else len(topic_freq) # The total number of topics (excluding the outlier topic -1)
        return total_topics  
    
    def getDocumentTopicWordsTopicScoresAndTopicIds(self,doc_ids):
        total_topic_words =[]
        total_word_scores = []
        topic_ids = []
        for doc_id in doc_ids: 
            topicId =  self.tweetIdTopicIdMappingDf[self.tweetIdTopicIdMappingDf [self.tweetIdColumn] ==doc_id].iloc[0][self.topicIdColumn]
            topic_words = self.topicModel.get_topic(topicId)
            current_topic_words = []
            current_word_scores=[]
            for word, score in topic_words:
                current_topic_words.append(word)
                current_word_scores.append(score)
            topic_ids.append(topicId)
            total_topic_words.append(current_topic_words)
            total_word_scores.append(current_word_scores)
        return total_topic_words,total_word_scores,topic_ids    
    
    
    



class Top2VecTopicExtractor(FeatureVectorMapper,AbstractTopicHeaderFinder,AbstractTopicExtractor):
   
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
    
    def getNumberOfTopics(self): 
        return self.topicModel.get_num_topics()
    
    def getTopicWordsScoresAndIds(self):
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
    
    def get_documents_topics(self, doc_ids,num_topics = 1):
        doc_topics, doc_dist, topic_words, topic_word_scores = self.topicModel.get_documents_topics(doc_ids,num_topics= num_topics)
        return doc_topics, doc_dist, topic_words, topic_word_scores
    
    
    def search_documents_by_topic(self,topic_num, num_docs):
        documents, document_scores, document_ids = self.topicModel.search_documents_by_topic(topic_num, num_docs)
        return documents, document_scores, document_ids
    
    
    def getWordVectorsOfWords(self,words):
        return self.topicModel._words2word_vectors(words)
    
    def getWordVectorsArray(self):
        return self.topicModel.word_vectors
    
    def getDocumentTopicWordsTopicScoresAndTopicIds(self,doc_ids):
        doc_topics, doc_dist, topic_words, topic_word_scores = self.get_documents_topics(doc_ids,num_topics = 1)
        return topic_words,topic_word_scores,doc_topics
    
    
        