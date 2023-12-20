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

modelpath =  DataDirHelper().getDataDir()+ "companyTweets\TopicModelAAPLFirst1000V2"
topicExtractor = Top2VecTopicExtractor(Top2VecTopicModelCreator().load(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\amazonTopicModelV2"))
topicsDf = pd.read_csv (DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\topicsChatGptNotCompound.csv")  
tokenizer = TweetTokenizer(DefaultWordFilter())
topicsCompare = LLMTopicsCompare(topicExtractor,tokenizer,topicsDf)

print("ChatGPT:")
for i in range(1,20):
    print(topicsCompare.calculateSimilarityScoreTop2Vec("topics_chat_gpt", i))