'''
Created on 09.02.2023

@author: vital
'''
from exploredata.TweetDataframeExplore import TweetDataframeExplore
import pandas as pd

class EqualClassSampler(object):
    '''
    Creates dataframe that has equal number of samples for each class
    '''


    def __init__(self, classColumnName = "class"):
        self.classColumnName = classColumnName
    
    def getDfWithEqualNumberOfClassSamples(self,df):
        value_counts = TweetDataframeExplore(df).getClassDistribution()  
        classWithLowestSamplesAmount = value_counts.iloc[len(value_counts)-1]
        resultDf = df[:0]
        for classLabel in value_counts.index:
            resultDf = pd.concat( [resultDf,df[df[self.classColumnName]== classLabel][: classWithLowestSamplesAmount]], ignore_index=True)
        return  resultDf
        