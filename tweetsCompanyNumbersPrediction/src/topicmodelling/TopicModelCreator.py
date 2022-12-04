'''
Created on 29.01.2022

@author: vital
'''

from top2vec import Top2Vec

class TopicModelCreator(object):
    '''
    classdocs
    '''
    def __init__(self,minVocabWordCount=None):
        if(minVocabWordCount is None):
            self.minVocabWordCount = 1
        else:
            self.minVocabWordCount = minVocabWordCount
        '''
        Constructor
        '''
    def createModel(self,sentences):
        #return Top2Vec(documents=sentences,embedding_model='universal-sentence-encoder',min_count=self.minVocabWordCount)
        return Top2Vec(documents=sentences,speed="learn", workers=8,min_count=self.minVocabWordCount)
    
    def load(self,path):
        return Top2Vec.load(path)
    