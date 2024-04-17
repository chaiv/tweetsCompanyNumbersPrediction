'''
Created on 16.04.2024

@author: vital
'''
import pandas as pd
from nlpvectors.TweetTokenizer import TweetTokenizer
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
from exploredata.POSTagging import PartOfSpeechTagging
from PredictionModelPath import AMAZON_REVENUE_10
from collections import Counter

tagger = PartOfSpeechTagging(TweetTokenizer(DefaultWordFilter()))

predictionModelPath =  AMAZON_REVENUE_10

tweetGroupsWithMostImportantWordsDf = pd.read_csv(
    predictionModelPath.getModelPath()+"\\tweetGroups_with_important_words_label_0.csv",
    sep='<'
    )


tweetGroupsWithMostImportantWordsDf = tweetGroupsWithMostImportantWordsDf[tweetGroupsWithMostImportantWordsDf["chatgpt_label"]==0.0]
pos_freq = Counter()
for index, row in tweetGroupsWithMostImportantWordsDf.iterrows():
    tokens = []
    for combinedToken in row["chatgpt_tokens"].split(","):
        tokensSplitOnSpace =combinedToken.split(" ")
        for token in tokensSplitOnSpace:
            tokens.append(token)
    for token in tokens: 
        posTag = tagger.getPOSTagOfToken(row["tweet_group"], token) 
        pos_freq.update([posTag.getPosTag()])
print(pos_freq)
    



#print(tweetGroupsWithMostImportantWordsDf["chatgpt_tokens"])
