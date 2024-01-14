'''
Created on 13.01.2024

@author: vital
'''
from nlpvectors.TweetGroup import TweetGroup
import pandas as pd

class TweetGroupToDataframe(object):
    '''
    Creates a simple dataframe to overview the content of tweet groups
    '''


    def __init__(self, 
                 tweetIdsColumn = "tweet_ids",
                 tweetSentencesColumn = "tweet_sentences",
                 tweetGroupLabelColumn = "label",
                 
                 ):
        pass
     
     
    def createTweetGroupDataframe(self, tweetGroups: list[TweetGroup]):   
        combinedSentencesIds = []
        labels = []
        tweetGroupIds = []
        combinedSentences = []
        for tweetGroup in tweetGroups: 
            combinedSentencesIds.append(";".join(tweetGroup.getSentenceIds()))
            labels.append(tweetGroup.getLabel())
            combinedSentences.append(";".join(tweetGroup.getSentences()))
        return pd.DataFrame(
            {
                
                
                
                
                }
            )