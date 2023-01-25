'''
Created on 15.01.2023

@author: vital
'''
import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
from pipeline.FeatureDataframePipeline import FeatureDataframePipeline
featuresClassesDf = FeatureDataframePipeline(). createDoc2VecFeaturesDf(pd.read_csv (DataDirHelper().getDataDir()+ 'companyTweets\\amazonTweetsWithNumbers.csv'),DataDirHelper().getDataDir()+'companyTweets\\amazonTopicModel')
print(featuresClassesDf)
featuresClassesDf.to_pickle (DataDirHelper().getDataDir()+ 'companyTweets\\featuresClassesAmazon.pkl')