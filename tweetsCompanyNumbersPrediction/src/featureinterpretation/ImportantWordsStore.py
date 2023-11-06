'''
Created on 17.02.2023

@author: vital
'''
import pandas as pd
import copy

        
def pad_dict_lists(input_dict):
    
    data = dict(input_dict)
    sublist_sizes = {}
    
    #calculate size of each sublist in a list value
    for key in data:
        dataForKey = data[key]
        if(isinstance(dataForKey, list)): 
            for i in range(len(dataForKey)):
                value = dataForKey[i]
                if isinstance(value, list):
                    sublist_sizes[i] = len(value)
                    
    if not sublist_sizes:
        return data
    
    #pad all other elements to lists of max value                                 
    for key in data:
        dataForKey = data[key]
        if(isinstance(data[key], list)): 
            for i in range(len(dataForKey)):
                if isinstance(dataForKey[i], list):
                    continue
                dataForKey[i] = [dataForKey[i]] * sublist_sizes[i]
        else: 
            data[key] = [dataForKey] * sum(value for value in sublist_sizes.values())
    return data

def flatten_dict_lists(data):
    flat_data = {}
    for key, value in data.items():
        if isinstance(value, list):
            if isinstance(value[0], list):
                flat_data[key] = [item for sublist in value for item in sublist]
            else: 
                flat_data[key] = value
    return flat_data



def createImportantWordStore(wordscoreWrappers,predictions):

    totalSentencesIds = []
    totalTokenIndexes = []
    totalTokens = []
    totalAttributions = []
    totalPredictions = []
  
    for wordscoreWrapperIndex  in range(len(wordscoreWrappers)):
        prediction = predictions[wordscoreWrapperIndex]
        wordscoreWrapper =wordscoreWrappers[wordscoreWrapperIndex] 
        wordscore_dict =  {
                "id" : wordscoreWrapper.getSentenceIds(),
                "token_index" : wordscoreWrapper.getTokenIndexes(),
                "token" : wordscoreWrapper.getTokens(),
                "attribution" : wordscoreWrapper.getAttributions(),
                "prediction" : prediction
                }
        
        wordscore_dict_flattened = flatten_dict_lists(pad_dict_lists(wordscore_dict))
        allLists = [wordscore_dict_flattened ["id"],wordscore_dict_flattened ["token_index"],wordscore_dict_flattened ["token"],wordscore_dict_flattened ["attribution"],wordscore_dict_flattened ["prediction"]]
        are_equal = all(len(lst) == len(allLists[0]) for lst in allLists)
        if are_equal:
            print("All lists have the same size.") 
        else: 
            print("Lists have different sizes.") 
        
        
        
        totalSentencesIds += wordscore_dict_flattened["id"]
        totalTokenIndexes += wordscore_dict_flattened["token_index"]
        totalTokens += wordscore_dict_flattened["token"]
        totalAttributions += wordscore_dict_flattened["attribution"]
        totalPredictions += wordscore_dict_flattened["prediction"]
        
        
    return ImportantWordStore(
        {
                "id" : totalSentencesIds,
                "token_index" : totalTokenIndexes,
                "token" : totalTokens,
                "attribution" : totalAttributions,
                "prediction" : totalPredictions
                }
        )
    



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