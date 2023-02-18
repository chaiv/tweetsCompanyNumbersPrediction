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
        
    def getDocumentVectorByTweetIndex(self, index):
        return self.vectorsToReturn[index]
    
    def getDocumentVectorsAsArray(self):
        return self.vectorsToReturn  
    
    def getFeatureVectorSize(self):
        return 3   