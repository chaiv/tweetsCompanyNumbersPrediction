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
from nlpvectors.TokenizerWordsLookup import TokenizedWordsLookup
from PredictionModelPath import AMAZON_REVENUE_10_LSTM_BINARY_CLASS
def main():
    #in main function to avoid multiprocessing errors
    #topicExtractor = Top2VecTopicExtractor(Top2VecTopicModelCreator().load(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\amazonTopicModelRandom15000"))
    topicExtractor = Top2VecTopicExtractor(Top2VecTopicModelCreator().load(DataDirHelper().getDataDir()+"companyTweets\\amazonTopicModel"))
    tweets = pd.read_csv (DataDirHelper().getDataDir()+ "companyTweets\\amazonTweetsWithNumbers.csv").head(15000)
    tweets.fillna('', inplace=True) #nan values in body columns 
    # topicExtractor = BertTopicExtractor(
    #     DataDirHelper().getDataDir()+ "companyTweets\\amazonTopicModelBert",
    #     TweetTokenizer(DefaultWordFilter()),
    #     pd.read_csv (DataDirHelper().getDataDir()+ "companyTweets\\amazonBertTopicMapping.csv")
    #     )
    topicEvaluation = TopicEvaluation(topicExtractor,TweetTokenizer(DefaultWordFilter()))
    total_topic_words,total_word_scores,topic_ids = topicExtractor.getTopicWordsScoresAndIds()
    print(topicExtractor.getNumberOfTopics())
    print("Topic diversity 10",topicEvaluation.get_topic_diversity(top_n=10))
    print("Topic diversity 20",topicEvaluation.get_topic_diversity(top_n=20))
    #print("Topic coherence",topicEvaluation.get_topic_coherence_with_extern_documents(tweets["body"].tolist()))
    tokenizerLookUp = TokenizedWordsLookup( AMAZON_REVENUE_10_LSTM_BINARY_CLASS.getModelPath()+"\\tokensDictionary.json")
    for topic_words in total_topic_words[:5]:
        for topic_word in topic_words:
            if(tokenizerLookUp.hasOriginalWord(topic_word)):
                print(tokenizerLookUp.getOriginalWord(topic_word))
        print("______________________")
if __name__ == '__main__':
    main()   
