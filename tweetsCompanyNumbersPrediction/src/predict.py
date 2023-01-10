'''
Created on 07.01.2023

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
from nlpvectors.tfidfVectorizer import TFIDFVectorizer
from tweetpreprocess.wordfiltering.StopWordsFilter import StopWordsFilter

numbersDfDateFormat='%d/%m/%Y %H:%M:%S'
tweets = pd.read_csv (r'C:\Users\vital\Google Drive\promotion\companyTweets\CompanyTweets.csv')
tweetsAmazon = TweetDataframeSorter(postTSPColumnName="post_date").sortByPostTSPAsc(TweetDataframeQuery().query(tweets, TweetQueryParams(companyName ="AMZN")))
numbersAmazon = pd.read_csv (r'C:\Users\vital\Google Drive\promotion\companyTweets\amazonQuarterRevenue.csv')
numbersDfWithTSP = DateToTimestampDataframeTransformer(dateToTSP=DateTSPConverter(dateFormat=numbersDfDateFormat)).addTimestampColumns(numbersAmazon)
tweetsWithNumbers = TweetNumbersConnector(postTSPColumn = "post_date").getTweetsWithNumbers(tweetsAmazon, numbersDfWithTSP)
#TODO lowercase filter and other unifier filters
textfiltetedTweetsWithNumbers  = TweetTextFilterTransformer(TextFilter([HyperlinkFilter(),StopWordsFilter()])).filterTextColumns(tweetsWithNumbers)  
tfidfVectorizer = TFIDFVectorizer(textfiltetedTweetsWithNumbers )
tweetsWithTFIDF = tfidfVectorizer.getTweetsWithTFIDFVectors()
tweetsWithTFIDF.to_csv(r'C:\Users\vital\Desktop\df\res.csv')

