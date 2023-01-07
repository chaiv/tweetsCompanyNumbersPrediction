'''
Created on 06.01.2023

@author: vital
'''

class TweetDataframeSorter(object):
 
    def __init__(self,postTSPColumnName = "post_tsp"):
        self.postTSPColumnName = postTSPColumnName
    
    def sortByPostTSPAsc(self,tweetsDf):
        return tweetsDf.sort_values(by=[self.postTSPColumnName])