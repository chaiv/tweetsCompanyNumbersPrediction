'''
Created on 15.01.2023

@author: vital
'''
import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
from topicmodelling.TopicExtractor import TopicExtractor
from topicmodelling.TopicModelCreator import TopicModelCreator
from nlpvectors.FeatureDataframeCreator import FeatureDataframeCreator
tweets = pd.read_csv (DataDirHelper().getDataDir()+ 'companyTweets\\amazonTweetsWithNumbers.csv')
modelpath =  r'C:\Users\vital\Desktop\df\amazonTopicModel'
mapper = TopicExtractor(TopicModelCreator().load(modelpath))
featuresClassesDf = FeatureDataframeCreator(mapper,classColumnName="value").createFeatureDataframe(tweets)
print(featuresClassesDf)
featuresClassesDf.to_csv (DataDirHelper().getDataDir()+ 'companyTweets\\FeaturesClassesAmazon.csv')