'''
Created on 18.12.2023

@author: vital
'''
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from tweetpreprocess.DataDirHelper import DataDirHelper
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
from nlpvectors.TweetTokenizer import TweetTokenizer


tweetsFile = "amazonTweetsWithNumbers.csv"
topicModelFile = "amazonTopicModelBertFirst1000"

#tweetsFile = "CompanyTweetsTeslaWithCarSales.csv"
#topicModelFile = "teslaTopicModel"
tweets = pd.read_csv (DataDirHelper().getDataDir()+ "companyTweets\\"+tweetsFile).head(1000)
tweets.fillna('', inplace=True) #nan values in body columns 
tokenizer = TweetTokenizer(DefaultWordFilter())
documents = []
for index, row in tweets.iterrows():
    documents.append(str(row["body"]))
vectorizer_model = CountVectorizer(tokenizer=tokenizer.tokenize)
topic_model = BERTopic(vectorizer_model=vectorizer_model,top_n_words=10)
topics, probs =topic_model.fit_transform(documents)
tweetIds = tweets["tweet_id"].to_list()
tweetIdTopicIdDf = pd.DataFrame({'tweet_id': tweetIds, 'topic_id': topics})
tweetIdTopicIdDf.to_csv(DataDirHelper().getDataDir()+ "companyTweets\\amazonBertTopicMappingFirst1000.csv")
topic_model.save(DataDirHelper().getDataDir()+ "companyTweets\\"+topicModelFile)