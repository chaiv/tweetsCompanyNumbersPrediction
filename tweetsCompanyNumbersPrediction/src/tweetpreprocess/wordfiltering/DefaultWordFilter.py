'''
Created on 05.03.2023

@author: vital
'''
import re
from tweetpreprocess.wordfiltering.AbstractTextFilter import AbstractTextFilter
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.porter import PorterStemmer

class DefaultWordFilter(AbstractTextFilter):
   
    def __init__(self, params):
        pass
    
    def filter(self, word):
        word = word.lower()
        word = re.sub(r'http\S+', '', word)
        # remove urls
        word = re.sub(r'http\S+', '', word)
        # remove mentions
        word = re.sub(r'@\w+', '', word)
        # remove hashtags
        word = re.sub(r'#\w+', '', word)
        word = remove_stopwords(word)
        # Remove punctuation
        translator = str.maketrans('', '', word.punctuation)
        word = word.translate(translator)
        # Remove digits
        word = re.sub('\d+', '', word)
        
        

        