'''
Created on 12.01.2023

@author: vital
'''
import pandas as pd
class FeatureDataframeCreator(object):
    '''
    classdocs
    '''


    def __init__(self, featureVectorMapper, tweetIdColumnName = "tweet_id", postTSPColumnName = "post_date", featuresColumnName="features", classColumnName="class"):
        self.featureVectorMapper = featureVectorMapper
        self.tweetIdColumnName = tweetIdColumnName
        self.postTSPColumnName = postTSPColumnName
        self.featuresColumnName = featuresColumnName
        self.classColumnName = classColumnName
    
    
    def createFeatureDataframe(self, tweetsWithClassesDf):
        featuresDf = tweetsWithClassesDf[[self.tweetIdColumnName,self.postTSPColumnName,self.classColumnName]]
        featuresDf[self.featuresColumnName] = pd.Series(self.featureVectorMapper.getFeatureVectorsAsArray())
        return featuresDf
        
           