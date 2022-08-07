'''
Created on 07.08.2022

@author: vital
'''
from tweetpreprocess.wordfiltering.AbstractTextFilter import AbstractTextFilter
class TextFilter(AbstractTextFilter):



    def __init__(self, subfilters):
        self.subfilters = subfilters
    
    def filter(self, text):
        for subfilter in self.subfilters: 
            text = subfilter.filter(text)
        return text    