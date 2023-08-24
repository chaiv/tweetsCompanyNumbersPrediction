'''
Created on 18.12.2022

@author: vital
'''
from tweetpreprocess.DateToTSP import DateTSPConverter

class TweetQueryParams(object):
 


    def __init__(self, companyName = None,companyNames = None, firstNTweets = None, tweetIds = None, fromDateStr = None, toDateStr = None,dateToTSP=DateTSPConverter()):
        self.companyName = companyName
        self.firstNTweets = firstNTweets
        self.tweetIds = tweetIds
        self.companyNames = companyNames
        if(fromDateStr is not None):
            self.fromDateTSP= dateToTSP.dateStrToTSPInt(fromDateStr)
        else:
            self.fromDateTSP = None
        if(toDateStr  is not None):
            self.toDateTSP = dateToTSP.dateStrToTSPInt(toDateStr)
        else: 
            self.toDateTSP = None

        