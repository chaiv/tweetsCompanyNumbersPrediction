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
        ax.text(daily_counts.index[-1], min_val, f'Min: {min_val}', color='blue', fontsize=10, va='center')
        ax.text(daily_counts.index[-1], max_val, f'Max: {max_val}', color='red', fontsize=10, va='center')
        ax.text(daily_counts.index[-1], average, f'Average: {average:.2f}', color='green', fontsize=10, va='center')
        plt.ylabel('Number of tweets')
        plt.title('Number of Tweets Per Day')
        plt.show()
        
    def createNumberOfWordsPlot(self):
        word_counts, min_val,max_val,average = self.dataframeExplore.getNumberOfWordsValues()
        plt.hist(word_counts, bins=20, color='c', edgecolor='black')
        plt.axvline(min_val, color='red', linestyle='dashed', linewidth=2)
        plt.axvline(average, color='green', linestyle='dashed', linewidth=2)
        plt.axvline(max_val, color='blue', linestyle='dashed', linewidth=2)
        plt.text(min_val,plt.ylim()[1] - 0.03 * plt.ylim()[1], f'Min: {min_val:.2f}', color='red', fontsize=10, va='center')
        plt.text(average,plt.ylim()[1] - 0.03 * plt.ylim()[1], f'Average: {average:.2f}', color='green', fontsize=10, va='center')
        plt.text(max_val,plt.ylim()[1] - 0.03 * plt.ylim()[1], f'Max: {max_val:.2f}', color='blue', fontsize=10, va='center')
        plt.xlabel('Number of words')
        plt.ylabel('Number of tweets')
        plt.title('Words per tweet distribution')
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
        plt.pie([total_tweets, near_duplicate_tweets], labels=["Total","Duplicates"], autopct='%1.1f%%', startangle=140)
        plt.axis('equal') 
        plt.title('Exact and Near Duplicate Tweets Percentage')
        plt.show() 
        
    def createExactDuplicatesPlot(self):
        plt.figure(figsize=(8, 8))
        total_tweets, near_duplicate_tweets = self.dataframeExplore.getExactDuplicateValues()
        plt.pie([total_tweets, near_duplicate_tweets], labels=["Total","Duplicates"], autopct='%1.1f%%', startangle=140)
        plt.axis('equal') 
        plt.title('Exact Duplicate Tweets Percentage')
        plt.show()    
        
        
    def createValPerTweetDatePlot(self,dateColumnDf, valColumnDf,min_val,max_val,average,xLabel,yLabel,title):
        plt.figure(figsize=(10, 6))
        plt.plot(dateColumnDf, valColumnDf, marker='o')
        plt.axhline(y=min_val, color='r', linestyle='--', label='Min Comment')
        plt.axhline(y=max_val, color='g', linestyle='--', label='Max Comment')
        plt.axhline(y=average, color='b', linestyle='--', label='Avg Comment')
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.title(title)
        plt.show() 
            
          
    def createCommentsPerTweetDatePlot(self):
        dateColumnDf, valColumnDf,min_val,max_val,average = self.dataframeExplore.getCommentValuesPerTweet()
        return self.createValPerTweetDatePlot(dateColumnDf, valColumnDf,min_val,max_val,average,'Tweet Date','Comment Number','Comment Number over Time')
    
    def createLikesPerTweetDatePlot(self):
        dateColumnDf, valColumnDf,min_val,max_val,average = self.dataframeExplore.getLikeValuesPerTweet()
        return self.createValPerTweetDatePlot(dateColumnDf, valColumnDf,min_val,max_val,average,'Tweet Date','Like Number','Like Number over Time')
    
    def createRetweetsPerTweetDatePlot(self):
        dateColumnDf, valColumnDf,min_val,max_val,average = self.dataframeExplore.getRetweetValuesPerTweet()
        return self.createValPerTweetDatePlot(dateColumnDf, valColumnDf,min_val,max_val,average,'Tweet Date','Retweet Number','Retweet Number over Time')
    
    def createTweetsPerWriterPlot(self):
        writer_counts, min_val,max_val,average = self.dataframeExplore.getTweetWritersCounts()
        plt.hist(writer_counts, bins=20, color='c', edgecolor='black')
        plt.axvline(min_val, color='red', linestyle='dashed', linewidth=2)
        plt.axvline(average, color='green', linestyle='dashed', linewidth=2)
        plt.axvline(max_val, color='blue', linestyle='dashed', linewidth=2)
        plt.legend({'Min':min_val, 'Average':average, 'Max':max_val})
        plt.title('Tweets per Writer')
        plt.xlabel("Number of tweets")
        plt.ylabel("Number of writers")
        plt.show() 
        
    def createPOSCountsPlot(self):
        pos_counts = self.dataframeExplore.getPOSCounts()
        plt.bar(list(pos_counts.keys()), list(pos_counts.values()), color='c', edgecolor='black')
        plt.xlabel('Parts of Speech')
        plt.ylabel('Counts')
        plt.title('Counts of Different Parts of Speech')
        plt.xticks(rotation=45, ha='right') 
        plt.show() 
    def createSentimentLabelNumbersPlot(self):
        _, ax = plt.subplots()
        self.dataframeExplore.getSentimentLabelsCounts().plot(kind='pie', ax=ax, autopct='%1.1f%%')
        ax.set_ylabel('') 
        plt.show()
        
    def createSentimentPolarityPerTweetDatePlot(self):
        dateColumnDf, valColumnDf,min_val,max_val,average = self.dataframeExplore.getSentimentPolarityPerTweet()
        return self.createValPerTweetDatePlot(dateColumnDf, valColumnDf,min_val,max_val,average,'Tweet Date','Sentiment Polarity','Sentiment Polarity over Time')

     
        

df =  pd.read_csv(DataDirHelper().getDataDir()+ 'companyTweets\\CompanyTweets.csv')
#df =  pd.read_csv(DataDirHelper().getDataDir()+ 'companyTweets\\CompanyTweetsAAPLFirst1000.csv')
dfGoogle = TweetDataframeQuery().query(df,TweetQueryParams(companyNames =["GOOG","GOOGL"]))
dfApple = TweetDataframeQuery().query(df,TweetQueryParams(companyNames =["AAPL"]))
dfMicrosoft = TweetDataframeQuery().query(df,TweetQueryParams(companyNames =["MSFT"]))
dfAmazon = TweetDataframeQuery().query(df,TweetQueryParams(companyNames =["AMZN"]))
dfTesla = TweetDataframeQuery().query(df,TweetQueryParams(companyNames =["TSLA"]))
print(len(dfGoogle),len(dfApple),len(dfMicrosoft),len(dfAmazon),len(dfTesla))
plots = DataframeExplorePlots(TweetDataframeExplore(df))
#plots.createCompanyTweetNumbersPlot()
#plots.createTweetsPerDayPlot() 
plots.createNumberOfWordsPlot()
#plots.createCardinalNumbersPlot()
#print(TweetDataframeExplore(df).getMostFrequentWordsNamedEntities(100))
#plots.createURLPerTweetsPlot()
#plots.createExactAndNearDuplicatesPlot()
#plots.createExactDuplicatesPlot()
#print(TweetDataframeExplore(df).printOriginalAndNearDuplicateRowsText())
#plots.createCommentsPerTweetDatePlot()
#plots.createLikesPerTweetDatePlot()
#plots.createRetweetsPerTweetDatePlot()
#print(TweetDataframeExplore(df).getMostFrequentWriters(100))
#plots.createTweetsPerWriterPlot()
#plots.createPOSCountsPlot()

#plots.createSentimentLabelNumbersPlot()
#plots.createSentimentPolarityPerTweetDatePlot()

