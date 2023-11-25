'''
Created on 24.11.2023

@author: vital
'''
import json
import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
from topicmodelling.TopicModelCreator import TopicModelCreator
from topicmodelling.TopicExtractor import TopicExtractor
from featureinterpretation.InterpretationDataframeUtil import addTopicOriginalWordsColumn
from tweetpreprocess.nearduplicates.NearDuplicateDetector import NearDuplicateDetector

topicExtractor = TopicExtractor(TopicModelCreator().load(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\amazonTopicModelV2"))
topic_words, word_scores, topic_nums = topicExtractor.topicModel.get_topics()
firstNtopics = 20
firstNTopicNums = topic_nums[:firstNtopics]
firstNTopicWords = topic_words[:firstNtopics]

topicNumColumn = "topic_num"
topicWordsColumn = "topic_words"
tweetColumn = "body"

dataDict = {
    topicNumColumn:[],
    tweetColumn :[],
    topicWordsColumn : []
    }
for i in range(0,len(firstNTopicNums)):
    topicNum = firstNTopicNums[i]
    topicWords = firstNTopicWords[i]
    documents, document_scores, document_ids= topicExtractor.search_documents_by_topic(topicNum,50)
    for doc in documents: 
        dataDict[topicNumColumn].append(topicNum )
        dataDict[tweetColumn].append(doc)
        dataDict[topicWordsColumn].append(topicWords )
firstNSentencesOfTopicsDf = pd.DataFrame(dataDict) 
with open(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\tokenizerLookupAmazon.csv") as json_file:
    tokenizerLookupDict= json.load(json_file)
addTopicOriginalWordsColumn(tokenizerLookupDict,firstNSentencesOfTopicsDf ,topicWordsColumnName=topicWordsColumn,originalTopicWordsColumnName = 'original_topic_words')
firstNSentencesOfTopicsDf = NearDuplicateDetector(firstNSentencesOfTopicsDf).getDataframeWithoutNearDuplicates()
firstNSentencesOfTopicsDf.to_csv(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\firstNtopicSentencesAmazon.csv")     

    
    
