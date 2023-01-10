'''
Created on 10.01.2023

@author: vital
'''
import re
from tweetpreprocess.wordfiltering.AbstractTextFilter import AbstractTextFilter

class LowercaseFilter(AbstractTextFilter):



    def __init__(self):
        '''
        Constructor
        '''
        
    def filter(self,text):  
        text = text.lower()
        return text
        
        