'''
Created on 18.02.2023

@author: vital
'''
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class AbstractTopicHeaderFinder():
    def getTopicHeaderByWord(self,searchWord):  
        pass
    def getTopicHeaderByIds(self,documentIds):
        pass
    

class TopicHeaderCalculator(object):

    def __init__(self):
        pass
    
    def calculateHeader(self,topicWords,topicWordVectors):
        if len(topicWords)==0 or len(topicWordVectors)==0:
            return None
        mean_vector = np.mean(topicWordVectors,axis=0)
        similarity_scores = [cosine_similarity([mean_vector], [vec])[0][0] for vec in topicWordVectors]
        return topicWords[np.argmax(similarity_scores)]
    
    