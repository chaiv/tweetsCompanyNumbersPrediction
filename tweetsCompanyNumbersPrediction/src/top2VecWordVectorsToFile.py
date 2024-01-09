'''
Created on 11.03.2023

@author: vital
'''
import numpy as np
from gensim.models import KeyedVectors
from tweetpreprocess.DataDirHelper import DataDirHelper
from topicmodelling.TopicModelCreator import Top2VecTopicModelCreator
from topicmodelling.TopicExtractor import Top2VecTopicExtractor
from nlpvectors.VocabularyCreator import PAD_TOKEN, UNK_TOKEN, SEP_TOKEN

modelpath =  DataDirHelper().getDataDir()+ "companyTweets\\teslaTopicModel"
wordVectorsPath = DataDirHelper().getDataDir()+ "companyTweets\\teslaWordVectors.txt"

# Load the word vectors
topicExtractor = Top2VecTopicExtractor(Top2VecTopicModelCreator().load(modelpath))
words = list(topicExtractor.getWordIndexes().keys())
word_vectors = topicExtractor.getWordVectorsArray()

# Add the PAD and UNK tokens
words.extend([PAD_TOKEN, UNK_TOKEN, SEP_TOKEN]) 
pad_vector = np.zeros((1, len(word_vectors[0])))  # PAD token has a zero vector
unk_vector = np.random.rand(1, len(word_vectors[0]))  # UNK token has a random vector
sep_vector = np.random.rand(1, len(word_vectors[0]))  # SEP token has a random vector
word_vectors = np.concatenate((word_vectors, pad_vector, unk_vector, sep_vector), axis=0)

# Create the KeyedVectors object and save to file
kv = KeyedVectors(len(word_vectors[0]))
kv.add_vectors(words, word_vectors)
kv.save_word2vec_format(wordVectorsPath, binary=False)
