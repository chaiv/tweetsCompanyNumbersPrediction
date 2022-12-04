'''
Created on 03.08.2022

@author: vital
'''
import pandas as pd
companyTweets = pd.read_csv (r'G:\Meine Ablage\promotion\companyTweets\CompanyTweets.csv')
company = 'AAPL'
companyTweetsApple = companyTweets[companyTweets['ticker_symbol']==company]
n = 100000
companyTweetsAppleFirstN = companyTweetsApple[:n]
companyTweetsAppleFirstN.to_csv(r'G:\Meine Ablage\promotion\companyTweets\CompanyTweets'+company+'First'+str(n)+'.csv')