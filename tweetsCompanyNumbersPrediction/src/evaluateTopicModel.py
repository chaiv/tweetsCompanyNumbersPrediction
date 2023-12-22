'''
Created on 05.12.2023

@author: vital
'''
import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
from topicmodelling.TopicModelCreator import Top2VecTopicModelCreator
from topicmodelling.TopicEvaluation import TopicEvaluation
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
from nlpvectors.TweetTokenizer import TweetTokenizer
from topicmodelling.TopicExtractor import Top2VecTopicExtractor,\
    BertTopicExtractor
def main():
    #in main function to avoid multiprocessing errors
    #topicExtractor = Top2VecTopicExtractor(Top2VecTopicModelCreator().load(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\amazonTopicModelRandom15000"))
    #topicExtractor = Top2VecTopicExtractor(Top2VecTopicModelCreator().load(DataDirHelper().getDataDir()+"companyTweets\\appleTopicModel"))
    tweets = pd.read_csv (DataDirHelper().getDataDir()+ "companyTweets\\amazonTweetsWithNumbers.csv").head(15000)
    tweets.fillna('', inplace=True) #nan values in body columns 
    topicExtractor = BertTopicExtractor(
        DataDirHelper().getDataDir()+ "companyTweets\\amazonTopicModelBertFirst15000",
        TweetTokenizer(DefaultWordFilter()),
        pd.read_csv (DataDirHelper().getDataDir()+ "companyTweets\\amazonBertTopicMappingFirst15000.csv")
        )
    topicEvaluation = TopicEvaluation(topicExtractor,TweetTokenizer(DefaultWordFilter()))
    print(topicExtractor.getNumberOfTopics())
    print("Topic diversity 10",topicEvaluation.get_topic_diversity(top_n=10))
    print("Topic diversity 20",topicEvaluation.get_topic_diversity(top_n=20))
    print("Topic coherence",topicEvaluation.get_topic_coherence_with_extern_documents(tweets["body"].tolist()))
if __name__ == '__main__':
    main()   
