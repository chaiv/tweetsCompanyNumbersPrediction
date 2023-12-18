from nlpvectors.TweetTokenizer import TweetTokenizer
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter

'''
Created on 12.11.2023

@author: vital
'''

import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
from featureinterpretation.InterpretationDataframeUtil import addUntokenizedWordColumnFromTweetDf,\
    addPOSTagsColumn, addTopicColumns, addTopicOriginalWordsColumn
from exploredata.POSTagging import PartOfSpeechTagging
from topicmodelling.TopicModelCreator import Top2VecTopicModelCreator
from topicmodelling.TopicExtractor import Top2VecTopicExtractor
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
importantWordsDf = addTopicColumns(
    Top2VecTopicExtractor(Top2VecTopicModelCreator().load(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\amazonTopicModelV2")),
                importantWordsDf,
                tweetIdColumnName, 
                topicNumColumnName, 
                topicWordsColumnName
                )

# importantWordsDf['topic_words'] = importantWordsDf['topic_words'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
# with open(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\tokenizerLookupAmazon.csv") as json_file:
#     tokenizerLookupDict= json.load(json_file)
# addTopicOriginalWordsColumn(tokenizerLookupDict,importantWordsDf,topicWordsColumnName='topic_words',originalTopicWordsColumnName = 'original_topic_words')

importantWordsDf[
    [
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