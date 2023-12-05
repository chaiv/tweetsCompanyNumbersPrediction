'''
Created on 05.12.2023

@author: vital
'''
from tweetpreprocess.DataDirHelper import DataDirHelper
from topicmodelling.TopicModelCreator import TopicModelCreator
from topicmodelling.TopicEvaluation import TopicEvaluation
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
from nlpvectors.TweetTokenizer import TweetTokenizer
from topicmodelling.TopicExtractor import TopicExtractor


modelpath =  DataDirHelper().getDataDir()+ "companyTweets\TopicModelAAPLFirst1000V2"
topicExtractor = TopicExtractor(TopicModelCreator().load(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\amazonTopicModelV2"))
topicEvaluation = TopicEvaluation(topicExtractor,TweetTokenizer(DefaultWordFilter()))
#print("Sihoutte score",topicEvaluation.get_silhoutte_score())
print("Topic coherence",topicEvaluation.get_topic_coherence())
print("Topic diversity",topicEvaluation.get_topic_diversity())