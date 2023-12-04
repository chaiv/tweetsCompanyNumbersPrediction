'''
Created on 17.04.2023

@author: vital
'''
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sklearn.metrics.pairwise import cosine_similarity
from nlpvectors.AbstractTokenizer import AbstractTokenizer
from topicmodelling.TopicExtractor import TopicExtractor

class TopicEvaluation(object):
    '''
    '''


    def __init__(self, topicModel : TopicExtractor,tokenizer:AbstractTokenizer):
        self.topicModel = topicModel
        self.tokenizer = tokenizer
    
    def get_topic_coherence(self):
        topic_words, _, _ =self.topicModel.get_topics()
        documents = [self.tokenizer.tokenize(doc) for doc in self.topicModel.get_documents()]
        dictionary = Dictionary(documents)
        cm = CoherenceModel(topics=topic_words, texts=documents, dictionary=dictionary)
        coherence = cm.get_coherence()
        return coherence
    
    def get_topic_diversity(self):
        topic_words, _, _ =self.topicModel.get_topics()
        
    
    
    
    def getCosineSimilarityMatrix(self):
        return cosine_similarity(self.topicModel.get_topic_vectors())
    
    
    
    
        