'''
Created on 05.12.2023

@author: vital
'''
from tweetpreprocess.DataDirHelper import DataDirHelper
from topicmodelling.TopicModelCreator import Top2VecTopicModelCreator
from topicmodelling.TopicEvaluation import TopicEvaluation
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
from nlpvectors.TweetTokenizer import TweetTokenizer
from topicmodelling.TopicExtractor import Top2VecTopicExtractor,\
    BertTopicExtractor
from bertopic import BERTopic
def main():
    #in main function to avoid multiprocessing errors
    #topicExtractor = Top2VecTopicExtractor(Top2VecTopicModelCreator().load(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\amazonTopicModelRandom15000"))
    #topicExtractor = Top2VecTopicExtractor(Top2VecTopicModelCreator().load(DataDirHelper().getDataDir()+"companyTweets\\appleTopicModel"))
    topicExtractor = BertTopicExtractor(BERTopic.load(DataDirHelper().getDataDir()+ "companyTweets\\amazonTopicModelBertRandom15000"))
    topicEvaluation = TopicEvaluation(topicExtractor,TweetTokenizer(DefaultWordFilter()))
    print(topicExtractor.getNumberOfTopics())
    #print("Sihoutte score",topicEvaluation.get_silhoutte_score())
    print("Topic coherence",topicEvaluation.get_topic_coherence())
    print("Topic diversity 10",topicEvaluation.get_topic_diversity(top_n=10))
    print("Topic diversity 20",topicEvaluation.get_topic_diversity(top_n=20))
if __name__ == '__main__':
    main()   
