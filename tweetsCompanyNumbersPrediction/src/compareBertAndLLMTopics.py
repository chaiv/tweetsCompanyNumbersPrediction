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


tokenizer = TweetTokenizer(DefaultWordFilter())
topicExtractor = BertTopicExtractor(
        DataDirHelper().getDataDir()+ "companyTweets\\amazonTopicModelBert",
        tokenizer,
        pd.read_csv (DataDirHelper().getDataDir()+ "companyTweets\\amazonBertTopicMapping.csv")
        )
topicsDf = pd.read_csv (DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\topicsChatGptNotCompound.csv")  
topicsCompare = LLMTopicsCompare(topicExtractor,tokenizer,topicsDf)
print("ChatGPT:")
print(topicsCompare.calculateSimilarityScoreBert("topics_chat_gpt"))
