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

def main():
    #in main function to avoid multiprocessing errors
    topicExtractor = TopicExtractor(TopicModelCreator().load(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\amazonTopicModelV2"))
    topicEvaluation = TopicEvaluation(topicExtractor,TweetTokenizer(DefaultWordFilter()))
    print(topicExtractor.getNumTopics())
    print("Sihoutte score",topicEvaluation.get_silhoutte_score())
    print("Topic coherence",topicEvaluation.get_topic_coherence())
    print("Topic diversity 10",topicEvaluation.get_topic_diversity(top_n=10))
    print("Topic diversity 20",topicEvaluation.get_topic_diversity(top_n=20))
if __name__ == '__main__':
    main()   
