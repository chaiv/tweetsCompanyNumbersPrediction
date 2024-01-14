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
                 tweetSentencesColumn = "tweet_sentences",
                 tweetGroupLabelColumn = "label"
                 ):
        self.tweetSentencesColumn = tweetSentencesColumn 
        self.tweetGroupLabelColumn = tweetGroupLabelColumn
     
     
    def createTweetGroupDataframe(self, tweetGroups: list[TweetGroup]):   
        labels = []
        combinedSentences = []
        for i in range(0,len(tweetGroups)):
            tweetGroup = tweetGroups[i] 
            labels.append(tweetGroup.getLabel())
            combinedSentences.append(";".join(tweetGroup.getSentences()))
        return pd.DataFrame(
            {
                self.tweetSentencesColumn : combinedSentences,
                self.tweetGroupLabelColumn : labels
                }
            )