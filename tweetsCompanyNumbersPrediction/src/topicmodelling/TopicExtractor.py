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
    
    def get_topics(self):
        return self.topicModel.get_topics()