'''
Created on 05.08.2023

@author: vital
'''
from datasketch import MinHash, MinHashLSH
import pandas as pd

df = pd.DataFrame(
                  [
                    ("I have two apples and 2 oranges at https://google.com"),
                    ("I have two apples and 2 oranges at https://google.com"),
                    ("I have two apples and 2 oranges at"),
                    ("Completely different tweet"),
                    ("I have two apples and 2 oranges at https://google.com")
                  ],
                  columns=["body"]
                  ) 

# Create MinHash objects
minhashes = {}
for c, i in enumerate(df['body']):
    minhash = MinHash(num_perm=128)
    for d in i:
        minhash.update(d.encode('utf8'))
    minhashes[c] = minhash

# Create LSH index
lsh = MinHashLSH(threshold=0.75, num_perm=128)
for i, minhash in minhashes.items():
    lsh.insert(i, minhash)

allDuplicateRowsIndexes = set()
# Find near duplicates
for i in range(len(df)):
    result = lsh.query(minhashes[i])
    if(len(result)>1 and i not in allDuplicateRowsIndexes):
        result.remove(i)
        for item in result:
            allDuplicateRowsIndexes.add(item)  
    print("Candidates with Jaccard similarity > 0.5 for input", i, ":", result)

print(allDuplicateRowsIndexes)
    
    