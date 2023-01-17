'''
Created on 15.01.2023

@author: vital
'''
from nlpvectors.FeatureVectorMapper import FeatureVectorMapper

class FeatureVectorMapperFake(FeatureVectorMapper):
    '''
    classdocs
    '''


    def __init__(self, vectorsToReturn):
        self.vectorsToReturn = vectorsToReturn
        
    def getFeatureVectorByTweetIndex(self, index):
        return self.vectorsToReturn[index]
    
    def getFeatureVectorsAsArray(self):
        return self.vectorsToReturn  
    
    def getFeatureVectorSize(self):
        return 3   