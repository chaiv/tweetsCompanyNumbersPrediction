'''
Created on 07.02.2023

@author: vital
'''
import pandas as pd
import spacy
from collections import Counter

class TweetDataframeExplore(object):



    def __init__(self, dataframe, 
                 bodyColumnName = "body",
                 classColumnName = "class",
                 companyNameColumn = "ticker_symbol",
                 postTSPColumn = "post_date"
                 ):
        self.dataframe = dataframe
        self.classColumnName = classColumnName
        self.bodyColumnName = bodyColumnName
        self.companyNameColumn = companyNameColumn
        self.postTSPColumn = postTSPColumn
        
    def getValueCounts(self,columnName):
        return  self.dataframe[columnName].value_counts()
    
    def getClassDistribution(self):
        return  self.getValueCounts(self.classColumnName)
    
    def getMostFrequentWords(self, firstN):  
        return  pd.Series(' '.join(self.dataframe[self.bodyColumnName].astype("string")).lower().split()).value_counts()[:firstN]
    
    def getCompanyTweetNumbers(self):
        return self.getValueCounts(self.companyNameColumn)
    
    def getTweetsPerDayValues(self):
        self.dataframe['date'] = pd.to_datetime(self.dataframe[self.postTSPColumn], unit='s')
        self.dataframe = self.dataframe.set_index('date')
        daily_counts = self.dataframe.resample('D').size()
        average = daily_counts.mean()
        max_val = daily_counts.max()
        min_val = daily_counts.min()
        return  daily_counts,min_val,max_val,average
    
    def getNumberOfWordsValues(self):
        self.dataframe['word_count'] = self.dataframe[self.bodyColumnName].apply(lambda x: len(str(x).split()))
        summary = self.dataframe['word_count'].describe()
        word_counts = self.dataframe['word_count']
        min_val = summary['min']
        max_val = summary['max']
        average = summary['mean']
        return word_counts, min_val,max_val,average
    
    def getMostFrequentWordsNamedEntities(self,firstN):
        nlp = spacy.load('en_core_web_sm')
        entity_freq = Counter()
        for batch in spacy.util.minibatch(self.dataframe[self.bodyColumnName], size=100):
            for doc in nlp.pipe(batch):
                entity_freq.update([entity.text for entity in doc.ents])
        return entity_freq.most_common(firstN)
    
    
        