'''
Created on 15.01.2023

@author: vital
'''
import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
from topicmodelling.TopicExtractor import TopicExtractor
from topicmodelling.TopicModelCreator import TopicModelCreator
from nlpvectors.FeatureDataframeCreator import FeatureDataframeCreator
tweets = pd.read_csv (DataDirHelper().getDataDir()+ 'companyTweets\\CompanyTweetsAAPLFirst1000WithNumbers.csv')
modelpath =  DataDirHelper().getDataDir()+ "companyTweets\TopicModelAAPLFirst1000"
mapper = TopicExtractor(TopicModelCreator().load(modelpath))
featuresClassesDf = FeatureDataframeCreator(mapper,classColumnName="class").createFeatureDataframe(tweets)
print(featuresClassesDf)
featuresClassesDf.to_csv (DataDirHelper().getDataDir()+ 'companyTweets\\FeaturesClassesAAPLFirst1000.csv')