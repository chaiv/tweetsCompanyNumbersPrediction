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
from classifier.ModelEvaluationHelper import loadModel
from classifier.transformer.Predictor import Predictor
from collections import Counter
from classifier.ClassificationMetrics import ClassificationMetrics
from featureinterpretation.ImportantWordsStore import createImportantWordStore
from featureinterpretation.AttributionsCalculator import AttributionsCalculator

manualTopic = "Climate Change"
topNImportantWords = 10
predictionModelPath = AMAZON_REVENUE_10_LSTM_BINARY_CLASS
predictionClassMapper = predictionModelPath.getPredictionClassMapper()
df = pd.read_csv(predictionModelPath.getDataframePath()) 
tokenizer = TweetTokenizer(DefaultWordFilter())
word_vectors = KeyedVectors.load_word2vec_format(predictionModelPath.getWordVectorsPath(), binary=False)
textEncoder = WordVectorsIDEncoder(word_vectors)
topicExtractor = BertTopicExtractor(
       DataDirHelper().getDataDir()+ "companyTweets\\amazonTopicModelBert",
       tokenizer,
       pd.read_csv (DataDirHelper().getDataDir()+ "companyTweets\\amazonBertTopicMapping.csv")
       )
fold = 0
modelPath = predictionModelPath.getModelPath()+"\\tweetpredict_fold"+str(fold)+".ckpt"
model = loadModel(modelPath,word_vectors,num_classes=predictionClassMapper.get_number_of_classes(),evalMode=False)
predictor = Predictor(model,tokenizer,textEncoder ,predictionClassMapper ,AttributionsCalculator(model,model.embedding))

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

prediction_classes = predictor.predictMultipleAsTweetGroupsInChunks(tweetGroups, 1000)
print("true_classes counts ",', '.join(f"{item}: {count}" for item, count in Counter(trueClasses).items()))
print("prediction_classes counts ",', '.join(f"{item}: {count}" for item, count in Counter(prediction_classes).items()))
metrics = ClassificationMetrics() 
print(metrics.classification_report(trueClasses, prediction_classes))
print("MCC "+str(metrics.calculate_mcc(trueClasses, prediction_classes)))
wordScoresWrappers = predictor.calculateWordScoresOfTweetGroupsInChunks(tweetGroups,observed_class=1,chunkSize=100,n_steps=500,internal_batch_size = 100)
importantWordsStore = createImportantWordStore(wordScoresWrappers,prediction_classes)
importantWordsDf = importantWordsStore.to_dataframe()
importantWordsDf= importantWordsDf.sort_values(by=["tweet_attribution", "token_attribution"], ascending=[False, False])
importantWordsDf= importantWordsDf.drop_duplicates(subset=["token"], keep="first")
topNImportantWords = list(zip(importantWordsDf["token"].head(topNImportantWords), importantWordsDf["token_attribution"].head(topNImportantWords)))
print(topNImportantWords)
