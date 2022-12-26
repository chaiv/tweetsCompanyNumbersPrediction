'''
Created on 26.12.2022

@author: vital
'''

class TweetTextFilterTransformer(object):

    def __init__(self, textFilter, tweetBodyColumn = 'body'):
        self.tweetBodyColumn = tweetBodyColumn
        self.textFilter = textFilter
        
    def filterTextColumns(self,tweetsDf):    
        tweetsDf[self.tweetBodyColumn] = tweetsDf.apply(lambda row: self.textFilter.filter(row[self.tweetBodyColumn]), axis=1)
        return  tweetsDf    