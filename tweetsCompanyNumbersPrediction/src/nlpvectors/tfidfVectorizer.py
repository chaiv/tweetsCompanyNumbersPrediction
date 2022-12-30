'''
Created on 30.01.2022

@author: vital
'''

from sklearn.feature_extraction.text import TfidfVectorizer 

class TFIDFVectorizer(object):
    '''
    Adds tfidf vectors to tweets dataframe
    '''


    def __init__(self,allTweetsDf,tfidfvectorsColumnName="tfidf"):
        self.tfidfvectorsColumnName = tfidfvectorsColumnName
        self.allTweetsDf = allTweetsDf
        self.tfidf_vectorizer=TfidfVectorizer(use_idf=True) 
        self.tfidf_vectorizer_vectors=self.tfidf_vectorizer.fit_transform(allTweetsDf['body'])
        
    def getTweetsWithTFIDFVectors(self):  
        allTweetsWithTFIDFVectorsDf = self.allTweetsDf.copy()
        allTweetsWithTFIDFVectorsDf[self.tfidfvectorsColumnName] = list(self.tfidf_vectorizer_vectors.toarray())
        return  allTweetsWithTFIDFVectorsDf       
    
    def getFeatureNames(self):
        return self.tfidf_vectorizer.get_feature_names_out()   