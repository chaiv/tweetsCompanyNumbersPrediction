from bertopic import BERTopic
from tweetpreprocess.DataDirHelper import DataDirHelper
from topicmodelling.TopicExtractor import BertTopicExtractor
topicModelFile = "amazonTopicModelBertRandom15000"
topicModel = BERTopic.load(DataDirHelper().getDataDir()+ "companyTweets\\"+topicModelFile)
topicExtractor = BertTopicExtractor(topicModel)
print(topicExtractor.getNumberOfTopics())
print(topicExtractor.getTopicWordsScoresAndIds())
