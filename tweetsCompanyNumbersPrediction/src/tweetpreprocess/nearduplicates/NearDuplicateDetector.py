'''
Created on 06.08.2023

@author: vital
'''

from datasketch import MinHash, MinHashLSH
import concurrent.futures

class NearDuplicateDetector(object):
    
    def __init__(self, dataframe, 
                 bodyColumnName = "body"):
        self.dataframe = dataframe
        self.bodyColumnName = bodyColumnName
        self.rowCounter = 0 
        self.chunkCounter = 0 
        
        
    def compute_minhash_row(self,row):
        minhash = MinHash(num_perm=128)
        for d in row:
            minhash.update(d.encode('utf8'))
        self.rowCounter+=1
        print("Row",self.rowCounter) 
        return minhash

    
    def getOriginalAndDuplicateRowsText(self,similarityThreshold=0.75):
        indexes_list = self.getOriginalRowsWithDuplicateRowIndexesDefault(similarityThreshold)
        texts = [self.dataframe[self.bodyColumnName].iloc[index] for index_list in indexes_list for index in index_list]
        return texts
    
    def getOriginalRowsWithDuplicateRowIndexesDefault(self,similarityThreshold=0.75):
        self.dataframe['minhash'] = self.dataframe[self.bodyColumnName].apply(self.compute_minhash_row)
        minhashes = self.dataframe['minhash'].to_dict()
        # Create LSH index
        lsh = MinHashLSH(threshold=similarityThreshold, num_perm=128)
        for i, minhash in minhashes.items():
            lsh.insert(i, minhash)
            
        allResults = []
        for i in range(len(self.dataframe)): 
            result = lsh.query(minhashes[i])
            allResults.append(result)
        return allResults
    
    def getDuplicateRowIndexesDefault(self,similarityThreshold=0.75): 
        allDuplicateRowsIndexes = set()
        # Find near duplicates
        for result in self.getOriginalRowsWithDuplicateRowIndexesDefault(similarityThreshold):
            if(len(result)>1 and result[0] not in allDuplicateRowsIndexes):
                result.remove(result[0])
                for item in result:
                    allDuplicateRowsIndexes.add(item)  
        return allDuplicateRowsIndexes
    
        
    def geDuplicateRowIndexes(self,similarityThreshold=0.75): 
        self.chunkCounter = 0  
        self.rowCounter = 0 
        return self.getDuplicateRowIndexesDefault(similarityThreshold)
        #return self.getDuplicateRowIndexesWithChunksParallel(similarityThreshold)