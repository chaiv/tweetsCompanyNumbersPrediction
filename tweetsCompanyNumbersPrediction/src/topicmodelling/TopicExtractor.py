'''
Created on 29.01.2022

@author: vital
'''



class TopicExtractor(object):
    '''
    classdocs
    '''


    def __init__(self, topicModel):
        self.topicModel = topicModel
      
      
    def getNumTopics(self): 
        return self.topicModel.get_num_topics()
    
    def get_topicwords(self):
        return self.topicModel.get_topics()[0]
    
    def searchTopics(self,keywords, num_topics):
        return self.topicModel.search_topics(keywords, num_topics)
    
    def search_documents_by_topic(self,topic_num, num_docs):
        return self.topicModel.search_documents_by_topic(topic_num, num_docs)
        