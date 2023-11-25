
import json
import pandas as pd
import ast
from tweetpreprocess.DataDirHelper import DataDirHelper
from featureinterpretation.InterpretationDataframeUtil import addTopicOriginalWordsColumn

importantWordsDf = pd.read_csv(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\importantWordsClass1AmazonSorted.csv")
importantWordsDf['topic_words'] = importantWordsDf['topic_words'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
with open(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\tokenizerLookupAmazon.csv") as json_file:
    tokenizerLookupDict= json.load(json_file)
addTopicOriginalWordsColumn(tokenizerLookupDict,importantWordsDf,topicWordsColumnName='topic_words',originalTopicWordsColumnName = 'original_topic_words')
importantWordsDf.to_csv(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\importantWordsClass1AmazonSorted.csv")
