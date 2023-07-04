'''
Created on 04.07.2023

@author: vital
'''
import pandas as pd
import matplotlib.pyplot as plt
from tweetpreprocess.DataDirHelper import DataDirHelper
from exploredata.TweetDataframeExplore import TweetDataframeExplore


df =  pd.read_csv(DataDirHelper().getDataDir()+ 'companyTweets\\CompanyTweets.csv')


pd.DataFrame(
                  [
                  ("a","AAPL"),
                  ("b","AAPL"),
                  ("c","TSLA")
                  ],
                  columns=["body","ticker_symbol"]
                  )

dfExplore = TweetDataframeExplore(df)

fig, ax = plt.subplots()

dfExplore.getCompanyTweetNumbers().plot(kind='pie', ax=ax, autopct='%1.1f%%')

ax.set_ylabel('')  # this removes the label

plt.show()