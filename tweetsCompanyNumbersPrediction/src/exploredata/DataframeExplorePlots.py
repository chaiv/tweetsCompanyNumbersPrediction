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
from datetime import timedelta


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
        daily_counts.plot(kind='line',color='black')
        ax.axhline(average, color='black', linestyle='--')
        ax.axhline(max_val, color='black', linestyle='--')
        ax.axhline(min_val, color='black', linestyle='--')
        ax.text(daily_counts.index[-1], min_val, f'Min: {min_val}', color='black', fontsize=16, fontname='Times New Roman', va='center')
        ax.text(daily_counts.index[-1], max_val, f'Max: {max_val}', color='black', fontsize=16, fontname='Times New Roman', va='center')
        ax.text(daily_counts.index[-1], average, f'Average: {average:.2f}', color='black', fontsize=16, fontname='Times New Roman', va='center')
        plt.ylabel('Number of tweets',fontsize=16, fontname='Times New Roman')
        #plt.title('Number of Tweets Per Day')
        plt.show()
        
        
    def createHistPlot(self,counts,min_val,max_val,average,xLabel,yLabel,title):
        plt.hist(counts, bins=20, color='lightgray', edgecolor='black')
        plt.axvline(min_val, color='black', linestyle='dashed', linewidth=2)
        plt.axvline(average, color='black', linestyle='dashed', linewidth=2)
        plt.axvline(max_val, color='black', linestyle='dashed', linewidth=2)
        y_max = plt.ylim()[1]
        offset = 0.05 * y_max  # Adjust the text position slightly above the line
        plt.text(min_val,y_max + offset, f'Min: {min_val:.2f}', color='black', fontsize=16, fontname='Times New Roman', va='center')
        plt.text(average,y_max + offset, f'Average: {average:.2f}', color='black', fontsize=16, fontname='Times New Roman', va='center')
        plt.text(max_val,y_max + offset, f'Max: {max_val:.2f}', color='black', fontsize=16, fontname='Times New Roman', va='center')
        plt.xlabel(xLabel,fontsize=16, fontname='Times New Roman')
        plt.ylabel(yLabel,fontsize=16, fontname='Times New Roman')
        #plt.title(title,fontsize=16, fontname='Times New Roman')
        plt.show()
        
        
    def createNumberOfLikesPlot(self):
        counts, min_val,max_val,average = self.dataframeExplore.getNumberOfLikes()
        return self.createHistPlot(counts,min_val,max_val,average,'Number of likes','Number of tweets','Likes per tweet distribution') 
    
    def createNumberOfCommentsPlot(self):
        counts, min_val,max_val,average = self.dataframeExplore.getNumberOfComments()
        return self.createHistPlot(counts,min_val,max_val,average,'Number of comments','Number of tweets','Comments per tweet distribution')     
    
    def createNumberOfRetweetsPlot(self):
        counts, min_val,max_val,average = self.dataframeExplore.getNumberOfRetweets()
        return self.createHistPlot(counts,min_val,max_val,average,'Number of retweets','Number of tweets','Retweets per tweet distribution')         
        
        
    def createNumberOfCharactersPlot(self):
        counts, min_val,max_val,average = self.dataframeExplore.getNumberofCharactersValues()
        return self.createHistPlot(counts,min_val,max_val,average,'Number of characters','Number of tweets','Characters per tweet distribution') 
        
    def createNumberOfWordsPlot(self):
        counts, min_val,max_val,average = self.dataframeExplore.getNumberOfWordsValues()
        return self.createHistPlot(counts,min_val,max_val,average,'Number of words','Number of tweets','Words per tweet distribution')
        
    def createWrittenNumbersPlot(self):
        counts, min_val,max_val,average = self.dataframeExplore.getWrittenNumbersPerTweetValues()
        return self.createHistPlot(counts,min_val,max_val,average,'Number of numbers','Number of tweets','Numbers per tweet distribution')

    
    def createURLPerTweetsPlot(self):
        counts, min_val,max_val,average = self.dataframeExplore.getURLPerTweetValues()
        return self.createHistPlot(counts,min_val,max_val,average,'Number of URLs','Number of tweets','URLs per tweet distribution')
 
        
    def createExactAndNearDuplicatesPlot(self):
        plt.figure(figsize=(8, 8))
        total_tweets, near_duplicate_tweets = self.dataframeExplore.getExactAndNearDuplicateValues()
        plt.pie([total_tweets, near_duplicate_tweets], labels=["Non-duplicates","Duplicates"], autopct='%1.1f%%', startangle=140)
        plt.axis('equal') 
        plt.title('Exact and near duplicate tweets percentage')
        plt.show() 
        
    def createExactDuplicatesPlot(self):
        plt.figure(figsize=(8, 8))
        total_tweets, near_duplicate_tweets = self.dataframeExplore.getExactDuplicateValues()
        plt.pie([total_tweets, near_duplicate_tweets], labels=["Non-duplicates","Duplicates"], autopct='%1.1f%%', startangle=140)
        plt.axis('equal') 
        plt.title('Exact duplicate tweets percentage')
        plt.show()    
        
        
    def createValPerTweetDatePlot(self,dateColumnDf, valColumnDf,min_val,max_val,average,xLabel,yLabel,title,
                                  xMinMaxAvrgLabelOffset = 100,yMinLabelOffset= -10,yMaxValOffset=0,yAvrgValOffset=0):
        plt.figure(figsize=(10, 6))
        plt.plot(dateColumnDf, valColumnDf, marker='o',color='black')
        plt.axhline(y=min_val, color='black', linestyle='--', label='Min Comment')
        plt.axhline(y=max_val, color='black', linestyle='--', label='Max Comment')
        plt.axhline(y=average, color='black', linestyle='--', label='Avg Comment')
        x_offset_days =xMinMaxAvrgLabelOffset
        x_offset_timedelta = timedelta(days=x_offset_days)
        plt.text(dateColumnDf.iloc[-1]+x_offset_timedelta, min_val+yMinLabelOffset, f'Min: {min_val:.2f}', color='black',fontsize=16, fontname='Times New Roman', va='center')
        plt.text(dateColumnDf.iloc[-1]+x_offset_timedelta, max_val+yMaxValOffset, f'Max: {max_val:.2f}', color='black', fontsize=16, fontname='Times New Roman', va='center')
        plt.text(dateColumnDf.iloc[-1]+x_offset_timedelta, average+yAvrgValOffset, f'Average: {average:.2f}', color='black', fontsize=16, fontname='Times New Roman', va='center')
        plt.xlabel(xLabel,fontsize=16, fontname='Times New Roman')
        plt.ylabel(yLabel,fontsize=16, fontname='Times New Roman')
        plt.title(title)
        plt.show() 
            
          
    def createCommentsPerTweetDatePlot(self):
        dateColumnDf, valColumnDf,min_val,max_val,average = self.dataframeExplore.getCommentValuesPerTweet()
        return self.createValPerTweetDatePlot(dateColumnDf, valColumnDf,min_val,max_val,average,'Tweet date','Comment number','Comment number over time')
    
    
    def createLikesPerTweetDatePlot(self):
        dateColumnDf, valColumnDf,min_val,max_val,average = self.dataframeExplore.getLikeValuesPerTweet()
        return self.createValPerTweetDatePlot(dateColumnDf, valColumnDf,min_val,max_val,average,'Tweet date','Like number','Like number over time')
    
    def createRetweetsPerTweetDatePlot(self):
        dateColumnDf, valColumnDf,min_val,max_val,average = self.dataframeExplore.getRetweetValuesPerTweet()
        return self.createValPerTweetDatePlot(dateColumnDf, valColumnDf,min_val,max_val,average,'Tweet date','Retweet number','Retweet number over time')
    
    def createTweetsPerWriterPlot(self):
        counts, min_val,max_val,average = self.dataframeExplore.getTweetWritersCounts()
        return self.createHistPlot(counts,min_val,max_val,average,'Number of tweets','Number of writers','Tweets per writer distribution')
        
    def createPOSCountsPlot(self):
        pos_counts = self.dataframeExplore.getPOSCounts()
        plt.bar(list(pos_counts.keys()), list(pos_counts.values()), color='lightgray', edgecolor='black')
        plt.xlabel('Parts of Speech')
        plt.ylabel('Counts')
        #plt.title('Counts of Different Parts of Speech')
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
df = dfApple
print(len(dfGoogle),len(dfApple),len(dfMicrosoft),len(dfAmazon),len(dfTesla))
dataframeExplore = TweetDataframeExplore(df)
plots = DataframeExplorePlots(dataframeExplore)


#plots.createSentimentLabelNumbersPlot()
#print(TweetDataframeExplore(dfMicrosoft).printOriginalAndNearDuplicateRowsText())
#plots.createPOSCountsPlot()
#dataframeExplore.printValueCounts(dataframeExplore.getMostFrequentWriters(100))
#plots.createExactAndNearDuplicatesPlot()
#print(TweetDataframeExplore(df).getMostFrequentWordsNamedEntities(100))
#print(TweetDataframeExplore(df).getMostFrequentNouns(100))
#print(TweetDataframeExplore(df).getMostFrequentVerbs(100))
#plots.createWrittenNumbersPlot()
#plots.createCompanyTweetNumbersPlot()
plots.createTweetsPerDayPlot() 
#plots.createNumberOfWordsPlot()
#plots.createNumberOfCharactersPlot()
#plots.createURLPerTweetsPlot()
#plots.createExactDuplicatesPlot()
#plots.createNumberOfLikesPlot()
#plots.createNumberOfCommentsPlot()
#plots.createNumberOfRetweetsPlot()
#plots.createTweetsPerWriterPlot()