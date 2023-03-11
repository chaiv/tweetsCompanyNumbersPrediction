

# from tweetpreprocess.DataDirHelper import DataDirHelper
# from topicmodelling.TopicExtractor import TopicExtractor
# from topicmodelling.TopicModelCreator import TopicModelCreator
# from nlpvectors.TweetTokenizer import TweetTokenizer
# from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
# modelpath =  DataDirHelper().getDataDir()+ "companyTweets\\amazonTopicModelV2"
# topicExtractor = TopicExtractor(TopicModelCreator().load(modelpath))
# topic_words,word_scores,topic_scores,topic_nums = topicExtractor.searchTopics(TweetTokenizer(DefaultWordFilter()).tokenize('Greek debt crisis'), 2)
# print(topic_words)

from gensim.models import KeyedVectors
from tweetpreprocess.DataDirHelper import DataDirHelper

# Load pre-trained token embeddings
word_vectors = KeyedVectors.load_word2vec_format(DataDirHelper().getDataDir()+ "companyTweets\WordVectorsAAPLFirst1000V2.txt", binary=False)

# List of tokens
tokens = ['aapl', 'banana', 'orange', 'pear', 'kiwi']


# List of indexes
indexes = [word_vectors.key_to_index[token] if token in word_vectors.key_to_index else -1 for token in tokens]

print(f"The indexes of {tokens} are {indexes}")
