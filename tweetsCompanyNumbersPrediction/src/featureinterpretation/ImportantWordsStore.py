'''
Created on 17.02.2023

@author: vital
'''
import pandas as pd
import itertools

        
def pad_dict_lists(data):
    transformedData = {}
    transformedData.update(data)
    max_sizes = {}
    for i in range(len(transformedData[list(transformedData.keys())[0]])):
        max_size = 0
        for key in transformedData:
            value = transformedData[key][i]
            if isinstance(value, list):
                max_size = max(max_size, len(value))
        for key in transformedData:
            max_sizes[key] = max_sizes.get(key, 0)
            if isinstance(transformedData[key][i], list):
                continue
            transformedData[key][i] = [transformedData[key][i]] * max_size
    return transformedData

def flatten_dict_lists(data):
    flat_data = {}
    for key, value in data.items():
        if isinstance(value, list):
            flat_data[key] = [item for sublist in value for item in sublist]
        else:
            flat_data[key] = value
    return flat_data




class ImportantWordStore:
    def __init__(self,data_dict):
        self.data_dict = data_dict

    def to_dataframe(self):
        transformedDict = flatten_dict_lists(pad_dict_lists(self.data_dict))
        return pd.DataFrame(transformedDict)
    
    def toDfWithFirstNSortedByAttribution(self,n,attributionColumnName = "attribution", ascending = True):
        df = self.to_dataframe()
        return self.toDfWithFirstNSortedByAttributionDfParam(n,df, attributionColumnName,ascending)
    
    def toDfWithFirstNSortedByAttributionDfParam(self,n,df,attributionColumnName, ascending):
        sorted_df = df.sort_values(attributionColumnName,ascending = ascending)
        return sorted_df.head(n)