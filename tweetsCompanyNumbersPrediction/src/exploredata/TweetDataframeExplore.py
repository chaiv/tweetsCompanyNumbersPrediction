'''
Created on 07.02.2023

@author: vital
'''
from collections import Counter
import re

import spacy

import pandas as pd
from sentiment.TweetSentimentAnalysis import TweetSentimentAnalysis
from tweetpreprocess.nearduplicates.DuplicateDetector import DuplicateDetector
from tweetpreprocess.nearduplicates.NearDuplicateDetector import NearDuplicateDetector


class TweetDataframeExplore(object):



    def __init__(self, dataframe, 
                 bodyColumnName = "body",
                 classColumnName = "class",
                 companyNameColumn = "ticker_symbol",
                 postTSPColumn = "post_date",
                 commentNumberColumn = "comment_num",
                 likeNumberColumn = "like_num",
                 retweetNumberColumn = "retweet_num",
                 writerColumn = "writer"
                 ):
        self.dataframe = dataframe
        self.classColumnName = classColumnName
        self.bodyColumnName = bodyColumnName
        self.companyNameColumn = companyNameColumn
        self.postTSPColumn = postTSPColumn
        self.commentNumberColumn = commentNumberColumn
        self.likeNumberColumn = likeNumberColumn
        self.retweetNumberColumn = retweetNumberColumn
        self.writerColumn = writerColumn
                
    def getValueCounts(self,columnName,dataframe):
        return  dataframe[columnName].value_counts()
    
    def getClassDistribution(self):
        return  self.getValueCounts(self.classColumnName,self.dataframe)
    
    def getMostFrequentWords(self, firstN):  
        return  pd.Series(' '.join(self.dataframe[self.bodyColumnName].astype("string")).lower().split()).value_counts()[:firstN]
    
    def getCompanyTweetNumbers(self):
        return self.getValueCounts(self.companyNameColumn,self.dataframe)
    
    def getTweetWritersCounts(self):
        writerCounts = self.getValueCounts(self.writerColumn,self.dataframe)
        average = writerCounts.mean()
        max_val = writerCounts.max()
        min_val = writerCounts.min()
        return  writerCounts,min_val,max_val,average
    
    def getMostFrequentWriters(self,firstN):
        writerCounts,_,_,_ = self.getTweetWritersCounts()
        return writerCounts.head(firstN)
    
    def getTweetsPerDayValues(self):
        self.dataframe['date'] = pd.to_datetime(self.dataframe[self.postTSPColumn], unit='s')
        self.dataframe = self.dataframe.set_index('date')
        daily_counts = self.dataframe.resample('D').size()
        average = daily_counts.mean()
        max_val = daily_counts.max()
        min_val = daily_counts.min()
        return  daily_counts,min_val,max_val,average
    
    def getNumberOfWordsValues(self):
        return self.countValuesPerTweet(self.bodyColumnName,self.dataframe,lambda x: len(str(x).split()))
    
    def getNumberofCharactersValues(self):
        return self.countValuesPerTweet(self.bodyColumnName,self.dataframe,lambda x: len(str(x)))

    def getNumberOfLikes(self):
        return self.countValuesPerTweet(self.likeNumberColumn,self.dataframe,lambda x: x)
    
    
    def getNumberOfComments(self):
        return self.countValuesPerTweet(self.commentNumberColumn,self.dataframe,lambda x: x)
    
    
    def getNumberOfRetweets(self):
        return self.countValuesPerTweet(self.retweetNumberColumn,self.dataframe,lambda x: x)


    def countValuesPerTweet(self,column,dataframe,countFunction):
        self.dataframe['count'] = dataframe[column].apply(countFunction)
        counts = self.dataframe['count']
        min_val =  counts.min()
        max_val =  counts.max()
        average =  counts.mean()
        return counts, min_val,max_val,average
        
    
    def getNamedEntities(self,documents):  
        nlp = spacy.load('en_core_web_sm', disable=['parser'])  
        docs_entities =[] 
        for doc in nlp.pipe(documents):
            docs_entities.append(doc.ents)
        return docs_entities        
        
    def getNamedEntitiesFrequences(self):
        nlp = spacy.load('en_core_web_sm', disable=['parser']) 
        entity_freq = Counter()
        i = 0 
        for doc in nlp.pipe(self.dataframe[self.bodyColumnName]):
            print(i := i + 1)
            entity_freq.update([entity.text for entity in doc.ents])
        return entity_freq
    
    def printValueCounts(self,valueCounts):
        for value, count in valueCounts.items():
            print(f"{value} : {count}")

    def getMostFrequentWordsNamedEntities(self,firstN):
        entity_freq = self.getNamedEntitiesFrequences()        
        return entity_freq.most_common(firstN)
    
    def getLeastFrequentWordNamedEntities(self,firstN):
        entity_freq = self.getNamedEntitiesFrequences()  
        return entity_freq.most_common()[:-firstN-1:-1]
    
    def count_numbers(self,documents):
        nlp = spacy.load('en_core_web_sm', disable=['parser']) 
        i = 0 
        counts = []
        for doc in nlp.pipe(documents):
            print(i := i + 1)
            counts.append(sum(1 for ent in doc.ents if ent.label_ == "CARDINAL"))
        return counts
    
    def getWrittenNumbersPerTweetValues(self):
        num_count = self.count_numbers(self.dataframe[self.bodyColumnName])
        self.dataframe['number_count'] = num_count   
        min_val = min(num_count)
        max_val = max(num_count)
        average = sum(num_count) / len(num_count)
        return num_count,min_val,max_val,average
    
    def count_urls_in_text(self,text):
        url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_regex, text)
        return len(urls)
    
    def getURLPerTweetValues(self):        
        return self.countValuesPerTweet(self.bodyColumnName,self.dataframe,self.count_urls_in_text)

    
    def getExactAndNearDuplicateValues(self):
        nearDuplicates = NearDuplicateDetector(self.dataframe,self.bodyColumnName).geDuplicateRowIndexes()
        totalTweetsNumber = len(self.dataframe)
        nearDuplicateTweetsNumber = len(nearDuplicates)
        return totalTweetsNumber,nearDuplicateTweetsNumber
    
    def getOriginalAndNearDuplicateRowsText(self):
        dfWithoutExactDuplicates = DuplicateDetector(self.dataframe,self.bodyColumnName).getDataframeWithoutDuplicates()
        return NearDuplicateDetector(dfWithoutExactDuplicates,self.bodyColumnName).getOriginalAndDuplicateRowsText()
    
    def printOriginalAndNearDuplicateRowsText(self):
        result = self.getOriginalAndNearDuplicateRowsText()
        for text_list in result:
            for text in text_list:
                print(text)
                print("")
            print("---")
    
    def getExactDuplicateValues(self):
        duplicatesDf = DuplicateDetector(self.dataframe,self.bodyColumnName).getDuplicatesDataframe()
        totalTweetsNumber = len(self.dataframe)
        exactDuplicatesTweetNumber= len(duplicatesDf)
        return totalTweetsNumber,exactDuplicatesTweetNumber
    
    def getColumnValuesPerTweetBasedOnDate(self,column,dataframe):
        dateColumnDf = pd.to_datetime(dataframe[self.postTSPColumn], unit='s')
        valColumnDf = dataframe[column]
        min_val = min(valColumnDf)
        max_val = max(valColumnDf)
        average = sum(valColumnDf) / len(valColumnDf)
        return dateColumnDf,valColumnDf, min_val,max_val,average
        
    def getCommentValuesPerTweet(self):
        return self.getColumnValuesPerTweetBasedOnDate(self.commentNumberColumn,self.dataframe)
    
    def getLikeValuesPerTweet(self):
        return self.getColumnValuesPerTweetBasedOnDate(self.likeNumberColumn,self.dataframe)
    
    def getRetweetValuesPerTweet(self):
        return self.getColumnValuesPerTweetBasedOnDate(self.retweetNumberColumn,self.dataframe)
    
    def getSentimentLabelsCounts(self):
        sentimentLabelColumnName_="sentiment_label"
        dfWithSentiment = TweetSentimentAnalysis(self.dataframe,bodyColumnName =self.bodyColumnName,sentimentPolarityColumnName="sentiment_polarity",sentimentLabelColumnName = sentimentLabelColumnName_ ).getDfWithSentiment()
        return self.getValueCounts(sentimentLabelColumnName_, dfWithSentiment)
    
    def getSentimentPolarityPerTweet(self):
        sentimentPolarityColumnName_ = "sentiment_polarity"
        dfWithSentiment = TweetSentimentAnalysis(self.dataframe,bodyColumnName =self.bodyColumnName,sentimentPolarityColumnName=sentimentPolarityColumnName_,).getDfWithSentiment()
        return self.getColumnValuesPerTweetBasedOnDate(sentimentPolarityColumnName_,dfWithSentiment)
    
    
    def getPOSCounts(self):
        nlp = spacy.load('en_core_web_sm') 
        pos_counts = {}
        i = 0 
        for doc in nlp.pipe(self.dataframe[self.bodyColumnName]):
            for token in doc:
                pos_type = token.pos_
                if pos_type in pos_counts:
                    pos_counts[pos_type] += 1
                else:
                    pos_counts[pos_type] = 1
            print(i := i + 1)    
        return pos_counts
        