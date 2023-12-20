import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
from bertopic import BERTopic
from topicmodelling.TopicExtractor import BertTopicExtractor

tweets = pd.read_csv (DataDirHelper().getDataDir()+ "companyTweets\\amazonTweetsWithNumbers.csv")
tweets.fillna('', inplace=True) #nan values in body columns 
topicExtractor = BertTopicExtractor(
        BERTopic.load(DataDirHelper().getDataDir()+ "companyTweets\\amazonTopicModelBert"),
        tweets
        )