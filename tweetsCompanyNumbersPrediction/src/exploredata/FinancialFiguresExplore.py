'''
Created on 19.09.2023

@author: vital
'''
import pandas as pd
class FinancialFiguresExplore(object):
    '''
    classdocs
    '''


    def __init__(self, dataframe,
                  fromDateColumnName = "from_date",
                  toDateColumnName="to_date",
                  valueColumnName = "value",
                  dateFormat = '%d/%m/%Y %H:%M:%S'
                  ):
        self.dataframe = dataframe
        self.fromDateColumnName = fromDateColumnName
        self.toDateColumnName = toDateColumnName
        self.valueColumnName = valueColumnName
        self.dateFormat = dateFormat
        
    def getDataframe(self):
        self.dataframe ['from_date'] = pd.to_datetime(self.dataframe[self.fromDateColumnName], format=self.dateFormat)
        self.dataframe ['to_date'] = pd.to_datetime(self.dataframe[self.toDateColumnName], format=self.dateFormat)
        return self.dataframe    