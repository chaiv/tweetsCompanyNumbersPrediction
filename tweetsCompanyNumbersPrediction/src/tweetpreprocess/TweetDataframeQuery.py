'''
Created on 18.12.2022

@author: vital
'''

class TweetDataframeQuery(object):


    def __init__(self,tweetDfCompanyColumn='ticker_symbol',tweetDfIdColumn='tweet_id',fromTSPColumn='from_date',toTSPColumn='to_date'):
        self.tweetDfCompanyColumn = tweetDfCompanyColumn
        self.tweetDfIdColumn = tweetDfIdColumn
        self.fromTSPColumn = fromTSPColumn
        self.toTSPColumn = toTSPColumn

        
        
        
        
    def query(self,allTweetsDf,queryParams):
        companyTweets = allTweetsDf
        if(queryParams.companyName is not None):
            companyTweets = companyTweets[companyTweets[self.tweetDfCompanyColumn]==queryParams.companyName]
        if(queryParams.tweetIds is not None):
            companyTweets = companyTweets[companyTweets[self.tweetDfIdColumn].isin(queryParams.tweetIds)]
        if(queryParams.fromDateTSP is not None):
            companyTweets = companyTweets[companyTweets[self.fromTSPColumn]>=queryParams.fromDateTSP]
        if(queryParams.toDateTSP is not None):
            companyTweets = companyTweets[companyTweets[self.toTSPColumn]<=queryParams.toDateTSP]
        if(queryParams.firstNTweets is not None): #Order important, first n tweets must be selected at the end!
            companyTweets = companyTweets[:queryParams.firstNTweets] 
        return companyTweets
        
        
        