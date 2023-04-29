

# from tweetpreprocess.DataDirHelper import DataDirHelper
# from topicmodelling.TopicExtractor import TopicExtractor
# from topicmodelling.TopicModelCreator import TopicModelCreator
# from nlpvectors.TweetTokenizer import TweetTokenizer
# from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
# modelpath =  DataDirHelper().getDataDir()+ "companyTweets\\amazonTopicModelV2"
# topicExtractor = TopicExtractor(TopicModelCreator().load(modelpath))
# topic_words,word_scores,topic_scores,topic_nums = topicExtractor.searchTopics(TweetTokenizer(DefaultWordFilter()).tokenize('Greek debt crisis'), 2)
# print(topic_words)

def split_list_on_indices(lst, indices):
    if not indices:
        return lst
    
    splitted_list = []
    start_idx = 0
    for idx in indices:
        sublist = lst[start_idx:idx]
        if sublist:
            splitted_list.append(sublist)
        start_idx = idx + 1
    sublist = lst[start_idx:]
    if sublist:
        splitted_list.append(sublist)

    return splitted_list

# Example usage:
lst = [0.1, 0.1, 0.0, 0.2, 0.2, 0.0]
indices = [2, 5]
splitted_list = split_list_on_indices(lst, indices)
print(splitted_list)  # Output: [[0.1, 0.1], [0.2, 0.2]]
