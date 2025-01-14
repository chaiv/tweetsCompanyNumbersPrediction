'''
Created on 13.01.2025

@author: vital
'''
import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
from nlpvectors.TweetTokenizer import TweetTokenizer
from topicmodelling.TopicExtractor import BertTopicExtractor
from PredictionModelPath import AMAZON_REVENUE_10_LSTM_BINARY_CLASS
from nlpvectors.DataframeSplitter import DataframeSplitter
from gensim.models.keyedvectors import KeyedVectors
from nlpvectors.WordVectorsIDEncoder import WordVectorsIDEncoder
from nlpvectors.TweetGroup import createTweetGroup

manualTopic = "Climate Change"

predictionModelPath = AMAZON_REVENUE_10_LSTM_BINARY_CLASS
df = pd.read_csv(predictionModelPath.getDataframePath()) 
tokenizer = TweetTokenizer(DefaultWordFilter())
word_vectors = KeyedVectors.load_word2vec_format(predictionModelPath.getWordVectorsPath(), binary=False)
textEncoder = WordVectorsIDEncoder(word_vectors)
topicExtractor = BertTopicExtractor(
       DataDirHelper().getDataDir()+ "companyTweets\\amazonTopicModelBert",
       tokenizer,
       pd.read_csv (DataDirHelper().getDataDir()+ "companyTweets\\amazonBertTopicMapping.csv")
       )


topic_words,word_scores,topic_scores,topic_nums  = topicExtractor.findTopics(manualTopic, 10)
document_ids = []
for topic_num in topic_nums:
    document_ids.extend(topicExtractor.get_document_ids_by_topic(topic_num, num_docs=10))
filtered_df = df[df["tweet_id"].isin(document_ids)]
tweetSplits = DataframeSplitter().getSplitIds(filtered_df,predictionModelPath.getTweetGroupSize())
tweetGroups = []
trueClasses = []
for tweetSplit in tweetSplits: 
    splitDf =  filtered_df[ filtered_df ["tweet_id"].isin( tweetSplit)]
    sentenceIds = tweetSplit
    sentences = splitDf ["body"].tolist()
    label = splitDf["class"].iloc[0]
    tweetGroup = createTweetGroup(tokenizer,textEncoder,sentences,sentenceIds ,label)
    tweetGroups.append(tweetGroup)
    trueClasses.append(tweetGroup.getLabel())

