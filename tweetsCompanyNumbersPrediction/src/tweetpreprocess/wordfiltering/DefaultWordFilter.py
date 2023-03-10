'''
Created on 05.03.2023

@author: vital
'''
import re
from tweetpreprocess.wordfiltering.AbstractTextFilter import AbstractTextFilter
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.porter import PorterStemmer
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import strip_tags
from attr.validators import min_len, max_len

class DefaultWordFilter(AbstractTextFilter):
   
    def __init__(self):
        self.stemmer = PorterStemmer()
    
    def filter(self, word):
        word = self.remove_underscore(word)
        word = self.convert_to_lowercase(word)
        word = self.remove_urls(word)
        word = self.remove_mentions(word)
        word = self.remove_hashtags(word)
        word = self.remove_stopwords(word)
        word = self.remove_punctuation(word)
        word = self.remove_digits(word)
        word = self.stem(word)
        word = self.simple_preprocess(word)
        return word

    def convert_to_lowercase(self, word):
        return word.lower()
    
    def remove_urls(self, word):
        return re.sub(r'http\S+', '', word)
    
    def remove_mentions(self, word):
        return re.sub(r'@', '', word)
    
    def remove_hashtags(self, word):
        return re.sub(r'#', '', word)
    
    def remove_stopwords(self, word):
        word = remove_stopwords(word)
        return word
    
    def remove_punctuation(self, word):
        return re.sub(r'[^\w\s]','', word)
    
    def remove_digits(self, word):
        return re.sub('\d+', '', word)
    
    def remove_underscore(self, word):
        return re.sub('_', '', word)
    
    def stem(self, word):
        return self.stemmer.stem(word)
        
    def simple_preprocess(self,word):    
        if(len(simple_preprocess(strip_tags(word),min_len= 3, deacc=True))>0):
            return word
        else:
            return ''
        