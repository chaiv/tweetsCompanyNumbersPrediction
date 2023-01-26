'''
Created on 15.01.2023

@author: vital
'''
import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
from pipeline.FeatureDataframePipeline import FeatureDataframePipeline
tweetsWithNumbersDfPath = DataDirHelper().getDataDir() + 'companyTweets\\amazonTweetsWithNumbers.csv'
#tweetsWithNumbersDfPath = DataDirHelper().getDataDir() + 'companyTweets\\CompanyTweetsAAPLFirst1000WithNumbers.csv'
topicModelPath = DataDirHelper().getDataDir() + 'companyTweets\\amazonTopicModel'
#topicModelPath = DataDirHelper().getDataDir() + 'companyTweets\\TopicModelAAPLFirst1000'
featuresClassesDf = FeatureDataframePipeline(). createDoc2VecFeaturesDf(pd.read_csv (tweetsWithNumbersDfPath),topicModelPath)
featuresClassesDf.to_pickle (DataDirHelper().getDataDir()+ 'companyTweets\\featuresClassesAmazon.pkl')
#featuresClassesDf.to_csv(DataDirHelper().getDataDir()+ 'companyTweets\\featuresClassesAAPLFirst1000.csv')