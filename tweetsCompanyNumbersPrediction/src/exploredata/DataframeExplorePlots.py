'''
Created on 04.07.2023

@author: vital
'''
import pandas as pd
import matplotlib.pyplot as plt
from tweetpreprocess.DataDirHelper import DataDirHelper
from exploredata.TweetDataframeExplore import TweetDataframeExplore
from tweetpreprocess.TweetDataframeQuery import TweetDataframeQuery
from tweetpreprocess.TweetQueryParams import TweetQueryParams


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
        
    def createNumberOfWordsPlot(self):
        word_counts, min_val,max_val,average = self.dataframeExplore.getNumberOfWordsValues()
        plt.hist(word_counts, bins=20, color='c', edgecolor='black')
        plt.axvline(min_val, color='red', linestyle='dashed', linewidth=2)
        plt.axvline(average, color='green', linestyle='dashed', linewidth=2)
        plt.axvline(max_val, color='blue', linestyle='dashed', linewidth=2)
        plt.legend({'Min':min_val, 'Average':average, 'Max':max_val})
        plt.show()
        
    def createCardinalNumbersPlot(self):
        word_counts, min_val,max_val,average = self.dataframeExplore.getCardinalNumbersPerTweetValues()
        plt.hist(word_counts, bins=20, color='c', edgecolor='black')
        plt.axvline(min_val, color='red', linestyle='dashed', linewidth=2)
        plt.axvline(average, color='green', linestyle='dashed', linewidth=2)
        plt.axvline(max_val, color='blue', linestyle='dashed', linewidth=2)
        plt.legend({'Min':min_val, 'Average':average, 'Max':max_val})
        plt.title('Cardinal numbers per tweet')
        plt.show()    
    
    def createURLPerTweetsPlot(self):
        url_counts, min_val,max_val,average = self.dataframeExplore.getURLPerTweetValues()
        plt.hist(url_counts, bins=20, color='c', edgecolor='black')
        plt.axvline(min_val, color='red', linestyle='dashed', linewidth=2)
        plt.axvline(average, color='green', linestyle='dashed', linewidth=2)
        plt.axvline(max_val, color='blue', linestyle='dashed', linewidth=2)
        plt.legend({'Min':min_val, 'Average':average, 'Max':max_val})
        plt.title('URLs per tweet')
        plt.show()    
        
    def createExactAndNearDuplicatesPlot(self):
        plt.figure(figsize=(8, 8))
        total_tweets, near_duplicate_tweets = self.dataframeExplore.getExactAndNearDuplicateValues()
        plt.pie([total_tweets, near_duplicate_tweets], labels=["Total","Near duplicates"], autopct='%1.1f%%', startangle=140)
        plt.axis('equal') 
        plt.title('Exact and Near Duplicate Tweets Percentage')
        plt.show() 
        
    def createExactDuplicatesPlot(self):
        plt.figure(figsize=(8, 8))
        total_tweets, near_duplicate_tweets = self.dataframeExplore.getExactDuplicateValues()
        plt.pie([total_tweets, near_duplicate_tweets], labels=["Total","Exact duplicates"], autopct='%1.1f%%', startangle=140)
        plt.axis('equal') 
        plt.title('Exact Duplicate Tweets Percentage')
        plt.show()    
          
        
        
# df = pd.DataFrame(
#                   [
#                   (1420070457,"First tweet","AAPL"),
#                   (1483230660,"Second tweet 2","AAPL"),
#                   (1483230660,"Third tweet 3 3","TSLA")
#                   ],
#                   columns=["post_date","body","ticker_symbol"]
#                   )

df =  pd.read_csv(DataDirHelper().getDataDir()+ 'companyTweets\\CompanyTweets.csv')
#df =  pd.read_csv(DataDirHelper().getDataDir()+ 'companyTweets\\CompanyTweetsAAPLFirst1000.csv')
df = TweetDataframeQuery().query(df,TweetQueryParams(companyName ="AMZN")).head(10000).reset_index(drop=True)
plots = DataframeExplorePlots(TweetDataframeExplore(df))
print(len(df))
#plots.createCompanyTweetNumbersPlot()
#plots.createTweetsPerDayPlot()
#plots.createNumberOfWordsPlot()
#plots.createCardinalNumbersPlot()
#print(TweetDataframeExplore(df).getMostFrequentWordsNamedEntities(100))
#plots.createURLPerTweetsPlot()
plots.createExactAndNearDuplicatesPlot()
#plots.createExactDuplicatesPlot()
print(TweetDataframeExplore(df).printOriginalAndNearDuplicateRowsText())