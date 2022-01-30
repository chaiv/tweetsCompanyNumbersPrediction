'''
Created on 29.01.2022

@author: vital
'''

from top2vec import Top2Vec

class TopicModelCreator(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
    def createModel(self,sentences):
        return Top2Vec(documents=sentences,speed="learn", workers=8)