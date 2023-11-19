from nlpvectors.TweetTokenizer import TweetTokenizer
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter

def addTopicsToDf(importantWordsDf, tweetIdColumnName, topicNumColumnName, topicWordsColumnName):
    topicExtractor = TopicExtractor(TopicModelCreator().load(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\amazonTopicModelV2"))
    doc_topics, doc_dist, topic_words, topic_word_scores = topicExtractor.get_documents_topics(importantWordsDf[tweetIdColumnName].tolist())
    importantWordsDf[topicNumColumnName] = doc_topics.tolist()
    importantWordsDf[topicWordsColumnName] = topic_words.tolist()

'''
Created on 12.11.2023

@author: vital
'''

import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
from featureinterpretation.ImportantWordsDataframeUtil import addUntokenizedWordColumnFromTweetDf,\
    addPOSTagsColumn
from exploredata.POSTagging import PartOfSpeechTagging
from topicmodelling.TopicModelCreator import TopicModelCreator
from topicmodelling.TopicExtractor import TopicExtractor
tweetDf = pd.read_csv(DataDirHelper().getDataDir()+ 'companyTweets\\amazonTweetsWithNumbers.csv')
importantWordsDf = pd.read_csv(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\importantWordsClass1Amazon.csv")
tweetIdColumnName = "tweet_id"
bodyColumnName = "body"
originaltokenColumnName = "original_token"
tokenAttributionColumnName = "token_attribution"
tweetAttributionColumnName = "tweet_attribution"
tweetPos = "tweet_pos"
posTagColumnName = "token_pos"
topicNumColumnName = "topic_num"
topicWordsColumnName = "topic_words"
importantWordsDf = importantWordsDf.sort_values(by=[tokenAttributionColumnName,tweetAttributionColumnName], ascending=False).head(100)
importantWordsDf = addUntokenizedWordColumnFromTweetDf(tweetDf,importantWordsDf)
importantWordsDf = addPOSTagsColumn(PartOfSpeechTagging(TweetTokenizer(DefaultWordFilter())),importantWordsDf)
addTopicsToDf(importantWordsDf, tweetIdColumnName, topicNumColumnName, topicWordsColumnName)
importantWordsDf[[
    originaltokenColumnName,
    posTagColumnName,
    bodyColumnName,
    tweetPos,
    tokenAttributionColumnName,
    tweetAttributionColumnName,
    topicNumColumnName,
    topicWordsColumnName 
    ]
    ].to_csv(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\importantWordsClass1AmazonSorted.csv")