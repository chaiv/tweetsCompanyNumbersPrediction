'''
Created on 18.12.2022

@author: vital
'''

class TweetDataframeQuery(object):


    def __init__(self,tweetDfCompanyColumn='ticker_symbol',tweetDfIdColumn='tweet_id',fromDateColumn='from_date',toDateColumn='to_date'):
        self.tweetDfCompanyColumn = tweetDfCompanyColumn
        self.tweetDfIdColumn = tweetDfIdColumn
        self.fromDateColumn = fromDateColumn
        self.toDateColumn = toDateColumn

        
        
        
        
    def query(self,allTweetsDf,queryParams):
        companyTweets = allTweetsDf
        if(queryParams.companyName is not None):
            companyTweets = companyTweets[companyTweets[self.tweetDfCompanyColumn]==queryParams.companyName]
        if(queryParams.tweetIds is not None):
            companyTweets = companyTweets[companyTweets[self.tweetDfIdColumn].isin(queryParams.tweetIds)]
        if(queryParams.fromDateTSP is not None):
            companyTweets = companyTweets[companyTweets[self.fromDateColumn]>=queryParams.fromDateTSP]
        if(queryParams.toDateTSP is not None):
            companyTweets = companyTweets[companyTweets[self.toDateColumn]<=queryParams.toDateTSP]
        if(queryParams.firstNTweets is not None): #Order important, first n tweets must be selected at the end!
            companyTweets = companyTweets[:queryParams.firstNTweets] 
        return companyTweets
        
        
        