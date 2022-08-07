'''
Created on 07.08.2022

@author: vital
'''

from gensim.parsing.preprocessing import remove_stopwords
from tweetpreprocess.wordfiltering.AbstractTextFilter import AbstractTextFilter
class StopWordsFilter(AbstractTextFilter):
    '''
    classdocs
    '''


    def filter(self,text):  
        return remove_stopwords(text)
        