'''
Created on 17.04.2023

@author: vital
'''
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sklearn.metrics.pairwise import cosine_similarity
from nlpvectors.AbstractTokenizer import AbstractTokenizer
from topicmodelling.TopicExtractor import AbstractTopicExtractor
from sklearn.metrics import silhouette_score

class TopicEvaluation(object):
    '''
    '''


    def __init__(self, topicModel : AbstractTopicExtractor,tokenizer:AbstractTokenizer):
        self.topicModel = topicModel
        self.tokenizer = tokenizer
    
    def get_topic_coherence(self):
        topic_words, _, _ =self.topicModel.getTopicWordsScoresAndIds()
        documents = [self.tokenizer.tokenize(doc) for doc in self.topicModel.get_documents()]
        dictionary = Dictionary(documents)
        #cm = CoherenceModel(topics=topic_words, corpus=corpus, dictionary=dictionary, coherence='u_mass') 
        cm = CoherenceModel(topics=topic_words, texts=documents, dictionary=dictionary)
        coherence = cm.get_coherence()
        return coherence
    
    def get_topic_diversity(self,top_n = 20):
        topic_words_list, _, _ =self.topicModel.getTopicWordsScoresAndIds()
        top_words_per_topic = []
        for topic_words in topic_words_list:
            top_words_per_topic.extend([word for word in topic_words[:top_n]])
        unique_words = set(top_words_per_topic)
        total_words = len(top_words_per_topic)
        topic_diversity = len(unique_words) / total_words
        return topic_diversity
        
    #def get_silhoutte_score(self):   
    #    return silhouette_score(self.topicModel.get_document_vectors(), self.topicModel.get_all_document_topics(), metric='cosine')
    
    
    
    #def getCosineSimilarityMatrix(self):
    #    return cosine_similarity(self.topicModel.get_topic_vectors())
    
    
    
    
        