'''
Created on 17.02.2023

@author: vital
'''
import pandas as pd
import copy

        
def pad_dict_lists(data):
    max_sizes = {}
    for i in range(len(data[list(data.keys())[0]])):
        max_size = 0
        for key in data:
            value = data[key][i]
            if isinstance(value, list):
                max_size = max(max_size, len(value))
        for key in data:
            max_sizes[key] = max_sizes.get(key, 0)
            if isinstance(data[key][i], list):
                continue
            data[key][i] = [data[key][i]] * max_size
    return data

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
        transformedDict = copy.deepcopy(self.data_dict)
        transformedDict = flatten_dict_lists(pad_dict_lists(transformedDict))
        return pd.DataFrame(transformedDict)
    
    def toDfWithFirstNSortedByAttribution(self,n,attributionColumnName = "attribution", ascending = True):
        df = self.to_dataframe()
        return self.toDfWithFirstNSortedByAttributionDfParam(n,df, attributionColumnName,ascending)
    
    def toDfWithFirstNSortedByAttributionDfParam(self,n,df,attributionColumnName, ascending):
        sorted_df = df.sort_values(attributionColumnName,ascending = ascending)
        return sorted_df.head(n)