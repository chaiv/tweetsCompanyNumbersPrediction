'''
Created on 05.08.2022

@author: vital
'''
import re
from tweetpreprocess.wordfiltering.AbstractTextFilter import AbstractTextFilter
class HyperlinkFilter(AbstractTextFilter):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
    
    
    def filter(self,text):  
        return re.sub(r'http\S+', '', text)