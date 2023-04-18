'''
Created on 17.04.2023

@author: vital
'''
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sklearn.metrics.pairwise import cosine_similarity
from nlpvectors.AbstractTokenizer import AbstractTokenizer

class TopicEvaluation(object):
    '''
    '''


    def __init__(self, topicModel,tokenizer:AbstractTokenizer):
        self.topicModel = topicModel
        self.tokenizer = tokenizer
    
    def get_topic_coherence(self):
        topic_words, _, _ =self.topicModel.get_topics()
        documents = [self.tokenizer.tokenize(doc) for doc in self.topicModel.documents]
        dictionary = Dictionary(documents)
        cm = CoherenceModel(topics=topic_words, texts=documents, dictionary=dictionary)
        coherence = cm.get_coherence()
        return coherence
    
    def getCosineSimilarityMatrix(self):
        return cosine_similarity(self.topicModel.topic_vectors)
    
    
    
    
        