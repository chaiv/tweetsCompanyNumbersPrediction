'''
Created on 10.01.2023

@author: vital
'''
import pandas as pd
from tweetpreprocess.TweetDataframeQuery import TweetDataframeQuery
from tweetpreprocess.TweetQueryParams import TweetQueryParams
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

numbersDfDateFormat='%d/%m/%Y %H:%M:%S'
tweets = pd.read_csv (DataDirHelper().getDataDir()+ "companyTweets\CompanyTweetsAAPLFirst1000.csv")
tweetsSubselect = TweetDataframeSorter(postTSPColumnName="post_date").sortByPostTSPAsc(TweetDataframeQuery().query(tweets, TweetQueryParams(companyName ="AAPL")))
numbers = pd.read_csv (DataDirHelper().getDataDir()+ "companyTweets\\amazonQuarterRevenue.csv")
numbersDfWithTSP = DateToTimestampDataframeTransformer(dateToTSP=DateTSPConverter(dateFormat=numbersDfDateFormat)).addTimestampColumns(numbers)
numbersWithClasses =  FiguresIncreaseDecreaseClassCalculator().getFiguresWithClasses(FiguresPercentChangeCalculator ().getFiguresWithClasses(numbersDfWithTSP))
tweetsWithNumbers = TweetNumbersConnector(postTSPColumn = "post_date",valueColumn="class").getTweetsWithNumbers(tweetsSubselect, numbersWithClasses)
textfiltetedTweetsWithNumbers  = TweetTextFilterTransformer(TextFilter([HyperlinkFilter()])).filterTextColumns(tweetsWithNumbers)  
print(textfiltetedTweetsWithNumbers )
textfiltetedTweetsWithNumbers.to_csv(DataDirHelper().getDataDir()+"companyTweets\CompanyTweetsAAPLFirst1000WithNumbers.csv")

