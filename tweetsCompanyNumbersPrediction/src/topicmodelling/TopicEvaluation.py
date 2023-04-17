'''
Created on 17.04.2023

@author: vital
'''
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sklearn.metrics.pairwise import cosine_similarity

class TopicEvaluation(object):
    '''
    classdocs
    '''


    def __init__(self, topicModel):
        self.topicModel = topicModel
    
    def get_topic_coherence(self):
        topic_words, _, _ =self.topicModel.get_topics()
        keys = list(self.topicModel.word_indexes.keys())
        key_list = [[key] for key in keys]
        dictionary = Dictionary(key_list)
        cm = CoherenceModel(topics=topic_words, texts=self.topicModel.documents, dictionary=dictionary)
        coherence = cm.get_coherence()
        return coherence
    
    def getCosineSimilarityMatrix(self):
        return cosine_similarity(self.topicModel.topic_vectors)
    
    
    
    
        