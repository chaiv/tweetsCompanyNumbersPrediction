'''
Created on 13.01.2025

@author: vital
'''
import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
from nlpvectors.TweetTokenizer import TweetTokenizer
from topicmodelling.TopicExtractor import BertTopicExtractor

manualTopic = "Climate Change"
topicExtractor = BertTopicExtractor(
       DataDirHelper().getDataDir()+ "companyTweets\\amazonTopicModelBert",
       TweetTokenizer(DefaultWordFilter()),
       pd.read_csv (DataDirHelper().getDataDir()+ "companyTweets\\amazonBertTopicMapping.csv")
       )
topic_words,word_scores,topic_scores,topic_nums  = topicExtractor.findTopics(manualTopic, 10)
print(topic_words)