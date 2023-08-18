'''
Created on 18.08.2023

@author: vital
'''
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
POSITIVE_LABEL = "Positive"
NEGATIVE_LABEL = "Negative"
NEUTRAL_LABEL = "Neutral"
class TweetSentimentAnalysis(object):
    '''
    classdocs
    '''


    def __init__(self,dataframe,bodyColumnName = "body",sentimentPolarityColumnName="sentiment_polarity",sentimentLabelColumnName="sentiment_label"):
        self.bodyColumnName = bodyColumnName
        self.sentimentPolarityColumnName = sentimentPolarityColumnName
        self.dataframe = dataframe
        self.sentimentLabelColumnName = sentimentLabelColumnName
        self.sia = SentimentIntensityAnalyzer()
    
    def get_sentiment_polarity(self,text):
        sentiment_scores = self.sia.polarity_scores(text)
        return sentiment_scores['compound']  
    
    def get_sentiment_label(self,polarity_score):
        if polarity_score > 0.05:
            return POSITIVE_LABEL
        elif polarity_score < -0.05:
            return NEGATIVE_LABEL
        else:
            return NEUTRAL_LABEL   
    
    def getDfWithSentiment(self):
        self.dataframe[self.sentimentPolarityColumnName] = self.dataframe[self.bodyColumnName].apply(self.get_sentiment_polarity)
        self.dataframe[self.sentimentLabelColumnName] = self.dataframe[self.sentimentPolarityColumnName].apply(self.get_sentiment_label)
        return self.dataframe