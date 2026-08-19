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
from nlpvectors.FeatureDataframeCreator import FeatureDataframeCreator
from tweetpreprocess.nearduplicates.DuplicateDetector import DuplicateDetector

class FeatureDataframePipeline(object):
    '''
    '''


    def __init__(self,
                 numbersDfDateFormat='%d/%m/%Y %H:%M:%S',
                 postTSPColumnName="post_date"
                 ):
        self.numbersDfDateFormat = numbersDfDateFormat
        self.postTSPColumnName = postTSPColumnName
    
    
    def createTweetWithNumbersDf(self,allTweetsDf,numbersDf,tweetQueryParams,classCalculator,
                                 removeExactDuplicates=False,removeNearDuplicates=False):
        tweetsSubselect = TweetDataframeSorter(postTSPColumnName=self.postTSPColumnName).sortByPostTSPAsc(TweetDataframeQuery().query(allTweetsDf,tweetQueryParams))
        # Duplicate tweets end up on both sides of any train/test split and let a model match test
        # texts verbatim. The data analysis measured 9.3% exact and 19.5% near duplicates, but the
        # removal was never part of this pipeline, so the archived labelled dataframes still contain
        # 13-16% exact duplicate bodies. Near-duplicate removal uses Simhash and is slow on millions
        # of rows, therefore it is optional.
        if removeExactDuplicates:
            rowsBefore = len(tweetsSubselect)
            tweetsSubselect = DuplicateDetector(tweetsSubselect).getDataframeWithoutDuplicates()
            print("Removed %d exact duplicate tweets of %d" % (rowsBefore - len(tweetsSubselect), rowsBefore))
        if removeNearDuplicates:
            # Simhash is an optional and comparatively expensive dependency, so import it only
            # when near-duplicate removal was requested.
            from tweetpreprocess.nearduplicates.NearDuplicateDetector import NearDuplicateDetector
            rowsBefore = len(tweetsSubselect)
            tweetsSubselect = NearDuplicateDetector(tweetsSubselect).getDataframeWithoutNearDuplicates()
            print("Removed %d near duplicate tweets of %d" % (rowsBefore - len(tweetsSubselect), rowsBefore))
        numbersDfWithTSP = DateToTimestampDataframeTransformer(dateToTSP=DateTSPConverter(dateFormat= self.numbersDfDateFormat)).addTimestampColumns(numbersDf)
        numbersWithClasses =  classCalculator.getFiguresWithClasses(numbersDfWithTSP)
        tweetsWithNumbers = TweetNumbersConnector(postTSPColumn = self.postTSPColumnName,valueColumn="class").getTweetsWithNumbers(tweetsSubselect, numbersWithClasses)
        return tweetsWithNumbers
    def createDoc2VecFeaturesDf(self, tweetsWithNumbersDf,topicModelPath):
        # Keep optional BERTopic/UMAP/Numba dependencies out of the financial-join pipeline.  Their
        # import can initialise compiled caches even when only duplicate filtering is requested.
        from topicmodelling.TopicExtractor import Top2VecTopicExtractor
        from topicmodelling.TopicModelCreator import Top2VecTopicModelCreator
        mapper = Top2VecTopicExtractor(Top2VecTopicModelCreator().load(topicModelPath))
        featuresDf = FeatureDataframeCreator(mapper,classColumnName="class").createFeatureDataframe(tweetsWithNumbersDf)
        return featuresDf
