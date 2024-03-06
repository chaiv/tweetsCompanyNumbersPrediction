'''
Created on 06.03.2024

@author: vital
'''
from classifier.LSTMNN import LSTMNN
from classifier.LSTMWithDropout import LSTMNNWithDropout

class CreateClassifierModel(object):



    def __init__(self, word_vectors):
        self.word_vectors = word_vectors
    
    def createModel(self):
        #return LSTMNNWithDropout(emb_size = 300,word_vectors = self.word_vectors)
        return LSTMNN(emb_size = 300,word_vectors = self.word_vectors)