'''
Created on 06.08.2023

@author: vital
'''

from datasketch import MinHash, MinHashLSH
import concurrent.futures

class NearDuplicateDetector(object):
    
    def __init__(self, dataframe, 
                 bodyColumnName = "body",similarityThreshold=0.95):
        self.dataframe = dataframe
        self.bodyColumnName = bodyColumnName
        self.rowCounter = 0 
        self.chunkCounter = 0 
        self.similarityThreshold = similarityThreshold
        
        
    def compute_minhash_row(self,row):
        minhash = MinHash(num_perm=128)
        for d in row:
            minhash.update(d.encode('utf8'))
        self.rowCounter+=1
        print("Row",self.rowCounter) 
        return minhash

    
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
    
    def getOriginalRowsWithDuplicateRowIndexesDefault(self):
        self.dataframe['minhash'] = self.dataframe[self.bodyColumnName].apply(self.compute_minhash_row)
        minhashes = self.dataframe['minhash'].to_dict()
        # Create LSH index
        lsh = MinHashLSH(threshold=self.similarityThreshold, num_perm=128)
        for i, minhash in minhashes.items():
            lsh.insert(i, minhash)
            
        allResults = []
        for i in range(len(self.dataframe)): 
            result = lsh.query(minhashes[i])
            allResults.append(result)
        return allResults
    
    def getDuplicateRowIndexesDefault(self): 
        allDuplicateRowsIndexes = set()
        # Find near duplicates
        for result in self.getOriginalRowsWithDuplicateRowIndexesDefault():
            if(len(result)>1 and result[0] not in allDuplicateRowsIndexes):
                result.remove(result[0])
                for item in result:
                    allDuplicateRowsIndexes.add(item)  
        return allDuplicateRowsIndexes
    
        
    def geDuplicateRowIndexes(self): 
        self.chunkCounter = 0  
        self.rowCounter = 0 
        return self.getDuplicateRowIndexesDefault()