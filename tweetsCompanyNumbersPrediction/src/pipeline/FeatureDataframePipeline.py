'''
Created on 24.01.2023

@author: vital
'''
from tweetpreprocess.TweetDataframeQuery import TweetDataframeQuery
from tweetpreprocess.TweetDataframeSorter import TweetDataframeSorter
from tweetpreprocess.DateToTSP import DateTSPConverter
from tweetpreprocess.DateToTimestampTransformer import DateToTimestampDataframeTransformer
from tweetnumbersconnector.tweetnumbersconnector import TweetNumbersConnector
from tweetpreprocess.wordfiltering.HyperlinkFilter import HyperlinkFilter
from tweetpreprocess.wordfiltering.TextFilter import TextFilter
from tweetpreprocess.TweetTextFilterTransformer import TweetTextFilterTransformer
from tweetpreprocess.DataDirHelper import DataDirHelper
from tweetpreprocess.FiguresIncreaseDecreaseClassCalculator import FiguresIncreaseDecreaseClassCalculator
from tweetpreprocess.FiguresPercentChangeCalculator import FiguresPercentChangeCalculator
from topicmodelling.TopicExtractor import Top2VecTopicExtractor
from topicmodelling.TopicModelCreator import Top2VecTopicModelCreator
from nlpvectors.FeatureDataframeCreator import FeatureDataframeCreator

class FeatureDataframePipeline(object):
    '''
    '''


    def __init__(self,
                 numbersDfDateFormat='%d/%m/%Y %H:%M:%S',
                 postTSPColumnName="post_date"
                 ):
        self.numbersDfDateFormat = numbersDfDateFormat
        self.postTSPColumnName = postTSPColumnName
    
    
    def createTweetWithNumbersDf(self,allTweetsDf,numbersDf,tweetQueryParams,classCalculator): 
        tweetsSubselect = TweetDataframeSorter(postTSPColumnName=self.postTSPColumnName).sortByPostTSPAsc(TweetDataframeQuery().query(allTweetsDf,tweetQueryParams))
        numbersDfWithTSP = DateToTimestampDataframeTransformer(dateToTSP=DateTSPConverter(dateFormat= self.numbersDfDateFormat)).addTimestampColumns(numbersDf)
        numbersWithClasses =  classCalculator.getFiguresWithClasses(numbersDfWithTSP)
        tweetsWithNumbers = TweetNumbersConnector(postTSPColumn = self.postTSPColumnName,valueColumn="class").getTweetsWithNumbers(tweetsSubselect, numbersWithClasses)
        return tweetsWithNumbers
    def createDoc2VecFeaturesDf(self, tweetsWithNumbersDf,topicModelPath):
        mapper = Top2VecTopicExtractor(Top2VecTopicModelCreator().load(topicModelPath))
        featuresDf = FeatureDataframeCreator(mapper,classColumnName="class").createFeatureDataframe(tweetsWithNumbersDf)
        return featuresDf
    
    
     
        