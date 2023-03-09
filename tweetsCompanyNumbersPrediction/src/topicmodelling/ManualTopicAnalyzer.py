'''
Created on 08.03.2023

@author: vital
'''

class ManualTopicAnalyzer(object):
    def __init__(self, topic_extractor,tokenColumn ="token",attributionColumn="attribution", topicIdColumn = "topicId"):
        self.topic_extractor = topic_extractor
        self.topicIdColumn = topicIdColumn
        self.tokenColumn = tokenColumn
        self.attributionColumn = attributionColumn
    
    def analyze(self, topics, important_words_df):
        relevant_words_dict = {}
        for topic in topics:
            _,_,_,topic_nums  = self.topic_extractor.searchTopics([topic], 1)
            topic_df = important_words_df[important_words_df[self.topicIdColumn] == topic_nums]
            relevant_words_list = []
            for _, row in topic_df.iterrows():
                relevant_words_list.append((row[self.tokenColumn], row[self.attributionColumn]))
            relevant_words_dict[topic] = relevant_words_list
        return relevant_words_dict
        