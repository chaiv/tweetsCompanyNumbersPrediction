# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 19:45:31 2021

@author: vital
"""
import pandas as pd
tweet = pd.read_csv ('C:\\Users\\vital\\Desktop\\Promotion\\companyTweets\\Tweet.csv')
companyTweet = pd.read_csv('C:\\Users\\vital\\Desktop\\Promotion\\companyTweets\\Company_Tweet.csv')
companyTweets = pd.merge(tweet,companyTweet,on='tweet_id')
companyTweets.to_csv('C:\\Users\\vital\\Desktop\\Promotion\\companyTweets\\CompanyTweetsFirst1000.csv')






