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

    tweetIdColumn = "tweet_id"
    tokenIndexColumn = "token_index"
    tokenColumn = "token"
    tokenAttributionColumn = "token_attribution"
    predictionColumn = "prediction"
    tweetAttributionColumn = "tweet_attribution"


    totalSentencesIds = []
    totalTokenIndexes = []
    totalTokens = []
    totalTokenAttributions = []
    totalPredictions = []
    totalTweetAttributions = []
  
    for wordscoreWrapperIndex  in range(len(wordscoreWrappers)):
        
        prediction = predictions[wordscoreWrapperIndex]
        wordscoreWrapper =wordscoreWrappers[wordscoreWrapperIndex] 
        wordscore_dict =  { #copy the lists because wordscoreWrapper should not be changed
                tweetIdColumn  : list(wordscoreWrapper.getSentenceIds()),
                tokenIndexColumn : list(wordscoreWrapper.getTokenIndexes()),
                tokenColumn : list(wordscoreWrapper.getTokens()),
                tokenAttributionColumn : list(wordscoreWrapper.getAttributions()),
                predictionColumn : prediction,
                tweetAttributionColumn: wordscoreWrapper.getAttributionsSum()
                }
        
        wordscore_dict_flattened = flatten_dict_lists(pad_dict_lists(wordscore_dict))
        totalSentencesIds += wordscore_dict_flattened[tweetIdColumn]
        totalTokenIndexes += wordscore_dict_flattened[tokenIndexColumn]
        totalTokens += wordscore_dict_flattened[tokenColumn]
        totalTokenAttributions += wordscore_dict_flattened[tokenAttributionColumn]
        totalPredictions += wordscore_dict_flattened[predictionColumn]
        totalTweetAttributions +=wordscore_dict_flattened[tweetAttributionColumn]
        
        
    return ImportantWordStore(
        {
                tweetIdColumn : totalSentencesIds,
                tokenIndexColumn : totalTokenIndexes,
                tokenColumn : totalTokens,
                tokenAttributionColumn : totalTokenAttributions,
                predictionColumn : totalPredictions,
                tweetAttributionColumn: totalTweetAttributions
                }
        )
    



class ImportantWordStore:
    def __init__(self,data_dict):
        self.data_dict = data_dict

    def to_dataframe(self):
        transformedDict = copy.deepcopy(self.data_dict)
        transformedDict = flatten_dict_lists(pad_dict_lists(transformedDict))
        return pd.DataFrame(transformedDict)
    
    def toDfWithFirstNSortedByAttribution(self,n,attributionColumnName = "token_attribution", ascending = True):
        df = self.to_dataframe()
        return self.toDfWithFirstNSortedByAttributionDfParam(n,df, attributionColumnName,ascending)
    
    def toDfWithFirstNSortedByAttributionDfParam(self,n,df,attributionColumnName, ascending):
        sorted_df = df.sort_values(attributionColumnName,ascending = ascending)
        return sorted_df.head(n)
