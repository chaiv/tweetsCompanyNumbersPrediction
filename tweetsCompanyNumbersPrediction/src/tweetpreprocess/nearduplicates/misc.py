'''
Created on 11.08.2023

@author: vital
'''
import pandas as pd
from simhash import Simhash, SimhashIndex
df = pd.DataFrame(
                  [
                    ("$AMZN closed my puts for breakeven avg, volcrush from earlier in the week got me. scratched trade."),
                    ("Not with $AMZN 's much superior margins. Analysts need to upgrade on this. And on flufflky toy sales, as well."),
                  ],
                  columns=["body"]
                  ) 
tweet_texts = df['body'].str.split()
simhash_values = [Simhash(text).value for text in tweet_texts]
index = SimhashIndex(zip(simhash_values), k=3) 
# Find near duplicates based on Hamming distance threshold
threshold = 10  # You can adjust this threshold based on your needs
near_duplicates = []
for i, simhash in enumerate(simhash_values):
    similar_hashes = index.get_near_dups(simhash)
    near_duplicates.extend([(i, j) for j in similar_hashes])

print("Near duplicate pairs:")
for pair in near_duplicates:
    print(df['tweets'][pair[0]])
    print(df['tweets'][pair[1]])
    print("-" * 20)