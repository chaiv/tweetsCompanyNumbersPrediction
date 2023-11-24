'''
Created on 24.11.2023

@author: vital
'''
import json
import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
from topicmodelling.TopicModelCreator import TopicModelCreator
from topicmodelling.TopicExtractor import TopicExtractor
from featureinterpretation.ImportantWordsDataframeUtil import addTopicOriginalWordsColumn

topicExtractor = TopicExtractor(TopicModelCreator().load(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\amazonTopicModelV2"))
topic_words, word_scores, topic_nums = topicExtractor.topicModel.get_topics()
firstNTopicNums = topic_nums[:5]
firstNTopicWords = topic_words[:5]
dataDict = {
    "topic_Num":[],
    "topic_words" : [],
    "tweet" :[]
    }
for i in range(0,len(firstNTopicNums)):
    topicNum = firstNTopicNums[i]
    topicWords = firstNTopicWords[i]
    documents, document_scores, document_ids= topicExtractor.search_documents_by_topic(topicNum,100)
    for doc in documents: 
        dataDict["topicNum"].append(topicNum )
        dataDict["topicWords"].append(topicWords )
        dataDict["tweet"].append(doc)
firstNSentencesOfTopicsDf = pd.DataFrame(dataDict) 
with open(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\tokenizerLookupAmazon.csv") as json_file:
    tokenizerLookupDict= json.load(json_file)
addTopicOriginalWordsColumn(tokenizerLookupDict,firstNSentencesOfTopicsDf ,topicWordsColumnName='topic_words',originalTopicWordsColumnName = 'original_topic_words')
print(firstNSentencesOfTopicsDf)       

    
    
