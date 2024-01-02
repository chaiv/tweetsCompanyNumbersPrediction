'''
Created on 22.12.2023

@author: vital
'''
import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
from topicmodelling.TopicModelCreator import Top2VecTopicModelCreator
from topicmodelling.TopicExtractor import Top2VecTopicExtractor,\
    BertTopicExtractor
from topicmodelling.llmcomparison.LLMTopicsCompare import LLMTopicsCompare
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
from nlpvectors.TweetTokenizer import TweetTokenizer
from gensim.models.keyedvectors import KeyedVectors
from nlpvectors.WordVectorsEncoder import WordVectorsEncoder


tokenizer = TweetTokenizer(DefaultWordFilter())
word_vectors = KeyedVectors.load_word2vec_format(DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsAmazonV2.txt", binary=False)
textEncoder = WordVectorsEncoder(word_vectors)
topicExtractor = BertTopicExtractor(
        DataDirHelper().getDataDir()+ "companyTweets\\amazonTopicModelBert",
        tokenizer,
        pd.read_csv (DataDirHelper().getDataDir()+ "companyTweets\\amazonBertTopicMapping.csv")
        )
#topicExtractor = Top2VecTopicExtractor(Top2VecTopicModelCreator().load(DataDirHelper().getDataDir()+ "companyTweets\\model\\amazonRevenueLSTMN5\\amazonTopicModelV2"))
topicsDf = pd.read_csv (DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\topicsChatGptNotCompound.csv")  
topicsCompare = LLMTopicsCompare(topicExtractor,tokenizer,textEncoder,topicsDf)
#print(topicsCompare.calculateSimilarityForExactTopicWords("topics_chat_gpt"))
for i in range(0,10):
    print(i,",",topicsCompare.calculateSimilarity("topics_chat_gpt",firstKTopics=i))
