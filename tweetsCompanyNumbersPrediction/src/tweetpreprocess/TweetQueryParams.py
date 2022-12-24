'''
Created on 18.12.2022

@author: vital
'''

class TweetQueryParams(object):
 


    def __init__(self, companyName = None, firstNTweets = None, tweetIds = None):
        self.companyName = companyName
        self.firstNTweets = firstNTweets
        self.tweetIds = tweetIds

        