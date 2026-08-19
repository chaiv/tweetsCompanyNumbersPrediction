'''
Created on 06.03.2024

@author: vital
'''
from classifier.LSTMNN import LSTMNN, LAST_HIDDEN_STATE_POOLING
from classifier.LSTMWithDropout import LSTMNNWithDropout

class CreateClassifierModel(object):



    def __init__(self, word_vectors, num_classes, class_weights=None,
                 pooling=LAST_HIDDEN_STATE_POOLING, padTokenIdx=None,
                 freeze_embeddings=False, lstmDropout=0.0):
        self.word_vectors = word_vectors
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.pooling = pooling
        self.padTokenIdx = padTokenIdx
        self.freeze_embeddings = freeze_embeddings
        self.lstmDropout = lstmDropout

    def createModel(self):
        #return LSTMNNWithDropout(emb_size = 300,word_vectors = self.word_vectors)
        return LSTMNN(emb_size = 300, word_vectors = self.word_vectors, num_classes = self.num_classes,
                      class_weights = self.class_weights, pooling = self.pooling,
                      padTokenIdx = self.padTokenIdx,
                      freeze_embeddings = self.freeze_embeddings,
                      lstmDropout = self.lstmDropout)
