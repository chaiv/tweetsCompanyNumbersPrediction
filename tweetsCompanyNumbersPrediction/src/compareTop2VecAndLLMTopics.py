'''
Created on 11.12.2023

@author: vital
'''
import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
from topicmodelling.TopicModelCreator import Top2VecTopicModelCreator
from topicmodelling.TopicExtractor import Top2VecTopicExtractor
from topicmodelling.llmcomparison.LLMTopicsCompare import LLMTopicsCompare
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
from nlpvectors.TweetTokenizer import TweetTokenizer

topicExtractor = Top2VecTopicExtractor(Top2VecTopicModelCreator().load(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\amazonTopicModelV2"))
tokenizer = TweetTokenizer(DefaultWordFilter())
topicsDf = pd.read_csv (DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\topicsChatGptNotCompound.csv")  
topicsCompare = LLMTopicsCompare(topicExtractor,tokenizer,topicsDf)
print("ChatGPT:")
#print(topicsCompare.calculateSimilarityScoreBert("topics_chat_gpt"))
for i in range(1,20):
    print(topicsCompare.calculateSimilarityScoreTop2Vec("topics_chat_gpt", i))
