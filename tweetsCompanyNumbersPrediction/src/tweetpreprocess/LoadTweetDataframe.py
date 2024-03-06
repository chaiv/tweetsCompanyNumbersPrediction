'''
Created on 05.03.2024

@author: vital
'''
import pandas as pd
from PredictionModelPath import PredictionModelPath
from tweetpreprocess.EqualClassSampler import EqualClassSampler

class LoadTweetDataframe(object):


    def __init__(self, predictionModelPath : PredictionModelPath):
        self.predictionModelPath = predictionModelPath
        
    def readDataframe(self):
        df = pd.read_csv(self.predictionModelPath.getDataframePath())
        if(self.predictionModelPath.hasEqualSamplesForEachClass()):
            return EqualClassSampler().getDfWithEqualNumberOfClassSamples(df)
        else: 
            return df
        
        
        