'''
Created on 07.02.2023

@author: vital
'''
import pandas as pd
class TweetDataframeExplore(object):



    def __init__(self, dataframe, bodyColumnName = "body",classColumnName = "class"):
        self.dataframe = dataframe
        self.classColumnName = classColumnName
        self.bodyColumnName = bodyColumnName
    
    def getClassDistribution(self):
        return  self.dataframe[ self.classColumnName].value_counts() 
    
    def getMostFrequentWords(self, firstN):  
        return  pd.Series(' '.join(self.dataframe[self.bodyColumnName]).lower().split()).value_counts()[:firstN]
    
    