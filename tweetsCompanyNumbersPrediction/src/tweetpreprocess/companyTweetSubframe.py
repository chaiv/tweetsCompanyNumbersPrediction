'''
Created on 03.08.2022

@author: vital
'''
import pandas as pd
companyTweets = pd.read_csv (r'G:\Meine Ablage\promotion\companyTweets\CompanyTweets.csv')
companyTweetsApple = companyTweets[companyTweets['ticker_symbol']=='AAPL']
companyTweetsAppleFirst1000 = companyTweetsApple[:1000]
companyTweetsAppleFirst1000.to_csv(r'G:\Meine Ablage\promotion\companyTweets\CompanyTweetsAppleFirst1000.csv')