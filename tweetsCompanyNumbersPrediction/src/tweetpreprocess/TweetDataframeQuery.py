'''
Created on 18.12.2022

@author: vital
'''

class TweetDataframeQuery(object):
    '''
    classdocs
    '''


    def __init__(self,tweetDfCompanyColumn='ticker_symbol',tweetDfIdColumn='tweet_id'):
        self.tweetDfCompanyColumn = tweetDfCompanyColumn
        self.tweetDfIdColumn = tweetDfIdColumn
        '''
        Constructor
        '''
        
        
        
        
    def query(self,allTweetsDf,queryParams):
        companyTweets = allTweetsDf
        if(queryParams.companyName is not None):
            companyTweets = companyTweets[companyTweets[self.tweetDfCompanyColumn]==queryParams.companyName]
        if(queryParams.firstNTweets is not None):
            companyTweets = companyTweets[:queryParams.firstNTweets] 
        if(queryParams.tweetIds is not None):
            companyTweets = companyTweets[companyTweets[self.tweetDfIdColumn].isin(queryParams.tweetIds)]
        return companyTweets
        
        
        