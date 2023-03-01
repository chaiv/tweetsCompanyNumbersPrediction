'''
Created on 17.02.2023

@author: vital
'''
import pandas as pd
import itertools

        
def getSublistSize(dic):
    for key, value in dic.items():
        if(type(value) == list and len(value)>0):
            firstElement = value[0]
            if(type(firstElement) == list):
                return len(value[0])
    return 0


def flattenList(l):
    return list(itertools.chain(*l))

def padAndFlattenDict(dic):
    paddedDict = {}
    sublistSize = getSublistSize(dic)
    if( sublistSize ==0):
        return dic
    for key, value in dic.items():
        if(type(value) == list and len(value)>0 and type(value[0]) != list):
            newValue = []
            for element in value:
                newValue.append([element]*sublistSize)
            paddedDict[key]= flattenList(newValue)
        else: 
            paddedDict[key]= flattenList(value)  
    return paddedDict




class ImportantWordStore:
    def __init__(self,data_dict):
        self.data_dict = data_dict

    def to_dataframe(self):
        transformedDict = padAndFlattenDict(self.data_dict)
        return pd.DataFrame(transformedDict)
    
    def toDfWithFirstNSortedByAttribution(self,n,attributionColumnName = "attribution", ascending = True):
        df = self.to_dataframe()
        return self.toDfWithFirstNSortedByAttributionDfParam(n,df, attributionColumnName,ascending)
    
    def toDfWithFirstNSortedByAttributionDfParam(self,n,df,attributionColumnName, ascending):
        sorted_df = df.sort_values(attributionColumnName,ascending = ascending)
        return sorted_df.head(n)