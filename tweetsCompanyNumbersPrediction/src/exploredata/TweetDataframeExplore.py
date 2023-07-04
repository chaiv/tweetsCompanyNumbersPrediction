'''
Created on 07.02.2023

@author: vital
'''
import pandas as pd
class TweetDataframeExplore(object):



    def __init__(self, dataframe, 
                 bodyColumnName = "body",
                 classColumnName = "class",
                 companyNameColumn = "ticker_symbol"
                 ):
        self.dataframe = dataframe
        self.classColumnName = classColumnName
        self.bodyColumnName = bodyColumnName
        self.companyNameColumn = companyNameColumn
        
    def getValueCounts(self,columnName):
        return  self.dataframe[columnName].value_counts()
    
    def getClassDistribution(self):
        return  self.getValueCounts(self.classColumnName)
    
    def getMostFrequentWords(self, firstN):  
        return  pd.Series(' '.join(self.dataframe[self.bodyColumnName].astype("string")).lower().split()).value_counts()[:firstN]
    
    def getCompanyTweetNumbers(self):
        return self.getValueCounts(self.companyNameColumn)
        