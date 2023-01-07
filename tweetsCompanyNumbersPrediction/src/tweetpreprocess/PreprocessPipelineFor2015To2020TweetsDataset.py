'''
Created on 06.01.2023

@author: vital
'''
from tweetpreprocess.DateToTSP import DateToTSP
from tweetpreprocess.DateToTimestampTransformer import DateToTimestampDataframeTransformer
from tweetnumbersconnector.tweetnumbersconnector import TweetNumbersConnector
from tweetpreprocess.wordfiltering.HyperlinkFilter import HyperlinkFilter
from tweetpreprocess.wordfiltering.TextFilter import TextFilter
from tweetpreprocess.TweetTextFilterTransformer import TweetTextFilterTransformer
from nlpvectors.tfidfVectorizer import TFIDFVectorizer

class PreprocessPipelineFor2015To2020TweetsDataset(object):

    def __init__(self):
        pass
     
     
    def createFilteredTweetsWithNumbers(self,tweetsDf,numbersDf):
        numbersDfDateFormat='%d/%m/%Y %H:%M:%S'
        numbersDfWithTSP = DateToTimestampDataframeTransformer(dateToTSP=DateToTSP(dateFormat=numbersDfDateFormat)).addTimestampColumns(numbersDf)
        tweetsWithNumbers = TweetNumbersConnector().getTweetsWithNumbers(tweetsDf, numbersDfWithTSP)
        textfiltetedTweetsWithNumbers  = TweetTextFilterTransformer(TextFilter([HyperlinkFilter()])).filterTextColumns(tweetsWithNumbers)  
        return  textfiltetedTweetsWithNumbers
        #TweetsWithNumbersWithVectors  = TFIDFVectorizer(textfiltetedTweetsWithNumbers).getTweetsWithTFIDFVectors()
        
        
        