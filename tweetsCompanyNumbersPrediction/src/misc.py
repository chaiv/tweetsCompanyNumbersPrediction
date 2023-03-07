

from tweetpreprocess.DataDirHelper import DataDirHelper
from topicmodelling.TopicExtractor import TopicExtractor
from topicmodelling.TopicModelCreator import TopicModelCreator
from nlpvectors.TweetTokenizer import TweetTokenizer
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
modelpath =  DataDirHelper().getDataDir()+ "companyTweets\\amazonTopicModelV2"
topicExtractor = TopicExtractor(TopicModelCreator().load(modelpath))
topic_words,word_scores,topic_scores,topic_nums = topicExtractor.searchTopics(TweetTokenizer(DefaultWordFilter()).tokenize('Greek debt crisis'), 2)
print(topic_words)
