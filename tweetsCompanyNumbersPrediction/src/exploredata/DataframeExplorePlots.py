'''
Created on 04.07.2023

@author: vital
'''
import pandas as pd
import matplotlib.pyplot as plt
from tweetpreprocess.DataDirHelper import DataDirHelper
from exploredata.TweetDataframeExplore import TweetDataframeExplore


class DataframeExplorePlots(object):
    
    def __init__(self, dataframeExplore: TweetDataframeExplore
                 ):
        self.dataframeExplore = dataframeExplore

    
    
    def createCompanyTweetNumbersPlot(self):
        _, ax = plt.subplots()
        self.dataframeExplore.getCompanyTweetNumbers().plot(kind='pie', ax=ax, autopct='%1.1f%%')
        ax.set_ylabel('') 
        plt.show()
        
    def createTweetsPerDayPlot(self):
        _, ax = plt.subplots()
        daily_counts,min_val,max_val,average = self.dataframeExplore.getTweetsPerDayValues()
        daily_counts.plot(kind='line')
        ax.axhline(average, color='green', linestyle='--')
        ax.axhline(max_val, color='red', linestyle='--')
        ax.axhline(min_val, color='blue', linestyle='--')
        ax.legend(["Daily Counts", "Average", "Max", "Min"])
        plt.ylabel('Number of tweets')
        plt.title('Number of Tweets Per Day')
        plt.show()
        
        
        
# df = pd.DataFrame(
#                   [
#                   (1420070457,"a","AAPL"),
#                   (1483230660,"b","AAPL"),
#                   (1483230660,"c","TSLA")
#                   ],
#                   columns=["post_date","body","ticker_symbol"]
#                   )

df =  pd.read_csv(DataDirHelper().getDataDir()+ 'companyTweets\\CompanyTweets.csv')

plots = DataframeExplorePlots(TweetDataframeExplore(df))

#plots.createCompanyTweetNumbersPlot()
plots.createTweetsPerDayPlot()