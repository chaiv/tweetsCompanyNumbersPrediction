# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 20:29:10 2021

@author: vital
"""
#import datetime
#import pandas as pd
#companyTweets = pd.read_csv('C:\\Users\\vital\\Desktop\\Promotion\\companyTweets\\CompanyTweets.csv')
#year2016tsp = int(datetime.datetime(2016,1,1,0,0).timestamp())
#amazon2015Tweets = companyTweets.loc[(companyTweets['ticker_symbol']=='AMZN') & (companyTweets['post_date']<year2016tsp)]
#print(len(amazon2015Tweets.index))


import datetime
import time
import pandas as pd
companyTweets = pd.read_csv(r'G:\Meine Ablage\promotion\companyTweets\CompanyTweetsDummyTesla.csv')
print(datetime.datetime.fromtimestamp(companyTweets.iloc[[0]]["post_date"]))
print(datetime.datetime.fromtimestamp(companyTweets.iloc[[1]]["post_date"]))
print(datetime.datetime.fromtimestamp(companyTweets.iloc[[2]]["post_date"]))


print(datetime.datetime.timestamp(datetime.datetime(2015, 1, 1)))
print(datetime.datetime.timestamp(datetime.datetime(2015, 3, 31)))
print(datetime.datetime.timestamp(datetime.datetime(2016, 1, 1,)))
print(datetime.datetime.timestamp(datetime.datetime(2016, 3, 31)))
print(datetime.datetime.timestamp(datetime.datetime(2017, 1, 1,)))
print(datetime.datetime.timestamp(datetime.datetime(2017, 3, 31)))

numbers = pd.read_csv(r'G:\Meine Ablage\promotion\companyTweets\numbersDummyTesla.csv')

postdate = int(companyTweets.iloc[[0]]["post_date"])
print(numbers.loc[(numbers ['from_date'] <= postdate) & (numbers['to_date'] >= postdate)]['value'])




