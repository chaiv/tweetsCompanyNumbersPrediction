'''
Created on 29.01.2022

@author: vital
'''
from nlpvectors.FeatureVectorMapper import FeatureVectorMapper



class TopicExtractor(FeatureVectorMapper):
    '''
    classdocs
    '''


    def __init__(self, topicModel):
        self.topicModel = topicModel
          
    def getFeatureVectorByTweetIndex(self,index):
        return self.topicModel.model.docvecs[index]    
    
    def getFeatureVectorByTweetId(self,tweetId):
        return self.topicModel.model.docvecs[self.topicModel.doc_id2index[tweetId]]
      
    def getFeatureVectorsAsArray(self):
        return self.topicModel.model.dv.vectors
    
    def getFeatureVectorSize(self):
        return self.topicModel.model.docvecs.vector_size
    
    def getNumTopics(self): 
        return self.topicModel.get_num_topics()
    
    def get_topics(self):
        topic_words, word_scores, topic_nums = self.topicModel.get_topics()
        return  topic_words, word_scores, topic_nums
    
    def searchTopics(self,keywords, num_topics):
        topic_words,word_scores,topic_scores,topic_nums  = self.topicModel.search_topics(keywords, num_topics)
        return topic_words,word_scores,topic_scores,topic_nums 
    
    def search_documents_by_topic(self,topic_num, num_docs):
        documents, document_scores, document_ids = self.topicModel.search_documents_by_topic(topic_num, num_docs)
        return documents, document_scores, document_ids
        