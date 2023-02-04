'''
Created on 04.02.2023

@author: vital
'''
import json 
from tweetpreprocess.DataDirHelper import DataDirHelper
from topicmodelling.TopicModelCreator import TopicModelCreator
from topicmodelling.TopicExtractor import TopicExtractor
modelpath =  DataDirHelper().getDataDir()+ "companyTweets\\amazonTopicModel"
dictionaryPath = DataDirHelper().getDataDir()+ "companyTweets\TokenizerAmazon.json"
topicExtractor = TopicExtractor(TopicModelCreator().load(modelpath))
with open(dictionaryPath, 'w') as f:
    f.write(json.dumps(topicExtractor.getWordIndexes()))

