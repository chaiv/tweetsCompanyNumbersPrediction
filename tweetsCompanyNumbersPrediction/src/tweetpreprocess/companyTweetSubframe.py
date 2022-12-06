'''
Created on 03.08.2022

@author: vital
'''
import pandas as pd
from tweetpreprocess.wordfiltering.HyperlinkFilter import HyperlinkFilter
from tweetpreprocess.wordfiltering.TextFilter import TextFilter
companyTweets = pd.read_csv (r'G:\Meine Ablage\promotion\companyTweets\CompanyTweets.csv')
company = 'AAPL'
companyTweetsApple = companyTweets[companyTweets['ticker_symbol']==company]
n = 1000
companyTweetsAppleFirstN = companyTweetsApple[:n]
companyTweetsAppleFirstN["body"] = companyTweetsAppleFirstN["body"].apply(lambda x: TextFilter([HyperlinkFilter()]).filter(x))
companyTweetsAppleFirstN.to_csv(r'G:\Meine Ablage\promotion\companyTweets\CompanyTweets'+company+'First'+str(n)+'.csv')