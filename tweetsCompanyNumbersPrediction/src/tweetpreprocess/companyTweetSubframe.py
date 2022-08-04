'''
Created on 03.08.2022

@author: vital
'''
import pandas as pd
companyTweets = pd.read_csv (r'G:\Meine Ablage\promotion\companyTweets\CompanyTweets.csv')
companyTweetsFirst1000 = companyTweets[:1000]
companyTweetsFirst1000.to_csv(r'G:\Meine Ablage\promotion\companyTweets\CompanyTweetsFirst1000.csv')