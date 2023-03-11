'''
Created on 11.03.2023

@author: vital
'''
import numpy as np
from gensim.models import KeyedVectors
from tweetpreprocess.DataDirHelper import DataDirHelper
from topicmodelling.TopicModelCreator import TopicModelCreator
from topicmodelling.TopicExtractor import TopicExtractor

modelpath =  DataDirHelper().getDataDir()+ "companyTweets\\amazonTopicModelV2"
wordVectorsPath = DataDirHelper().getDataDir()+ "companyTweets\WordVectorsAmazonV2.txt"

# Load the word vectors
topicExtractor = TopicExtractor(TopicModelCreator().load(modelpath))
words = list(topicExtractor.getWordIndexes().keys())
word_vectors = topicExtractor.getWordVectorsArray()

# Add the PAD and UNK tokens
pad_token = len(words)  # Index of PAD token is the length of words
unk_token = len(words) + 1  # Index of UNK token is one more than the length of words
words.extend(['PAD', 'UNK'])  # Add PAD and UNK to the list of words
pad_vector = np.zeros((1, len(word_vectors[0])))  # PAD token has a zero vector
unk_vector = np.random.rand(1, len(word_vectors[0]))  # UNK token has a random vector
word_vectors = np.concatenate((word_vectors, pad_vector, unk_vector), axis=0)

# Create the KeyedVectors object and save to file
kv = KeyedVectors(len(word_vectors[0]))
kv.add_vectors(words, word_vectors)
kv.save_word2vec_format(wordVectorsPath, binary=False)
