'''
Created on 06.08.2023

@author: vital
'''
import re
from simhash import Simhash, SimhashIndex

class NearDuplicateDetector(object):
    
    def __init__(self, dataframe, 
                 bodyColumnName = "body"):
        self.dataframe = dataframe
        self.bodyColumnName = bodyColumnName

            
    
    def getOriginalAndDuplicateRowsText(self):
        indexes_list = self.getOriginalRowsWithDuplicateRowIndexesDefault()
        tuple_indexes_list = [tuple(index_list) for index_list in indexes_list]
        unique_tuple_indexes_list = list(set(tuple_indexes_list))
        unique_indexes_list = [list(index_tuple) for index_tuple in unique_tuple_indexes_list]         # because rows are cross referencing there are duplicate indexes, they must be removed
        only_duplicates_indexes_list = [index_list for index_list in unique_indexes_list if len(index_list) > 1]         # also remove row indexes that has no duplicates
        
        texts_lists = []
        for index_list in only_duplicates_indexes_list:
            text_list = [self.dataframe[self.bodyColumnName].iloc[index] for index in index_list]
            texts_lists.append(text_list)     
        return texts_lists
    
    def get_features(self,s):
        width = 3
        s = s.lower()
        s = re.sub(r'[^\w]+', '', s)
        return [s[i:i + width] for i in range(max(len(s) - width + 1, 1))]    
    
    def getOriginalRowsWithDuplicateRowIndexesDefault(self):
        tweet_texts = self.dataframe[self.bodyColumnName].tolist()
        simhashes = []
        for i in range(len(tweet_texts)):
            tweet_text = tweet_texts[i]
            val = (i,Simhash(self.get_features(tweet_text)))
            simhashes.append(val)
            print("Row "+str(i))
        index = SimhashIndex(simhashes, k=3)
        allResults = []
        for i in range(len(self.dataframe)): 
            result = index.get_near_dups(simhashes[i][1])
            result = [int(item) for item in result]
            result = sorted(result)
            allResults.append(result)
        return allResults
        
        
    def geDuplicateRowIndexes(self): 
        allDuplicateRowsIndexes = set()
        for result in self.getOriginalRowsWithDuplicateRowIndexesDefault():
            if(len(result)>1 and result[0] not in allDuplicateRowsIndexes):
                result.remove(result[0])
                for item in result:
                    allDuplicateRowsIndexes.add(item)  
        return allDuplicateRowsIndexes