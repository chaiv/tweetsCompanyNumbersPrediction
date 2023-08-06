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
        self.chunkCounter = 0 
        
        
    def compute_minhash_row(self,row):
        minhash = MinHash(num_perm=128)
        for d in row:
            minhash.update(d.encode('utf8'))
        return minhash

    def compute_minhash_batch(self,chunk):
        chunk_hashes =[self.compute_minhash_row(row) for row in chunk]
        self.chunkCounter+=1
        print("Chunk",self.chunkCounter) 
        return chunk_hashes
    
    
    def getDuplicateRowIndexesDefault(self,similarityThreshold=0.75): 
        self.dataframe['minhash'] = self.dataframe[self.bodyColumnName].apply(self.compute_minhash_row)
        minhashes = self.dataframe['minhash'].to_dict()
        # Create LSH index
        lsh = MinHashLSH(threshold=similarityThreshold, num_perm=128)
        for i, minhash in minhashes.items():
            lsh.insert(i, minhash)
        
        allDuplicateRowsIndexes = set()
        # Find near duplicates
        for i in range(len(self.dataframe)):
            result = lsh.query(minhashes[i])
            if(len(result)>1 and i not in allDuplicateRowsIndexes):
                result.remove(i)
                for item in result:
                    allDuplicateRowsIndexes.add(item)  
        return allDuplicateRowsIndexes
    
    def getDuplicateRowIndexesWithChunksParallel(self,similarityThreshold=0.75): 
        self.chunkCounter = 0       
        self.dataframe.fillna('', inplace=True) #nan values in body columns    
        input_list = self.dataframe[self.bodyColumnName].tolist()
        print("Rows",len(input_list))
        # Define the number of chunks
        num_chunks = 100
        
        # Split the input list into chunks        
        chunk_size = len(input_list) // num_chunks
        if len(input_list) % num_chunks != 0:
            chunk_size += 1
        
        chunks = [input_list[i:i+chunk_size] for i in range(0, len(input_list), chunk_size)]
        print("Total chunks",len(chunks))

        # Process each chunk in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            minhashes_list = list(executor.map(self.compute_minhash_batch, chunks))

        # Flatten the list of MinHashes
        minhashes = {}
        for c, minhashes_chunk in enumerate(minhashes_list):
            for i, minhash in enumerate(minhashes_chunk):
                minhashes[c * chunk_size + i] = minhash
                
        lsh = MinHashLSH(threshold=similarityThreshold, num_perm=128)
        for i, minhash in minhashes.items():
            lsh.insert(i, minhash)
        
        allDuplicateRowsIndexes = set()
        # Find near duplicates
        for i in range(len(self.dataframe)):
            result = lsh.query(minhashes[i])
            if(len(result)>1 and i not in allDuplicateRowsIndexes):
                result.remove(i)
                for item in result:
                    allDuplicateRowsIndexes.add(item)  
        return allDuplicateRowsIndexes
    
        
    def geDuplicateRowIndexes(self,similarityThreshold=0.75): 
        #return self.getDuplicateRowIndexesDefault(similarityThreshold)
        return self.getDuplicateRowIndexesWithChunksParallel(similarityThreshold)