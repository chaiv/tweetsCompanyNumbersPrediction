'''
Created on 04.02.2023

@author: vital
'''
import json 
from tweetpreprocess.DataDirHelper import DataDirHelper
from topicmodelling.TopicModelCreator import TopicModelCreator
from topicmodelling.TopicExtractor import TopicExtractor
#modelpath =  DataDirHelper().getDataDir()+ "companyTweets\\TopicModelAAPLFirst1000V2"
#dictionaryPath = DataDirHelper().getDataDir()+ "companyTweets\VocabularyAAPLFirst1000V2.json"
modelpath =  DataDirHelper().getDataDir()+ "companyTweets\\amazonTopicModelV2"
dictionaryPath = DataDirHelper().getDataDir()+ "companyTweets\VocabularyAmazonV2.json"
topicExtractor = TopicExtractor(TopicModelCreator().load(modelpath))
with open(dictionaryPath, 'w') as f:
    f.write(json.dumps(topicExtractor.getWordIndexes()))

