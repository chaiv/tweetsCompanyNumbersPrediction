'''
Created on 08.03.2023

@author: vital
'''
import pandas as pd
import unittest
from topicmodelling.ManualTopicAnalyzer import ManualTopicAnalyzer

class TestManualTopicAnalyzer(unittest.TestCase):

    def test_analyze(self):
        # Test analyzing one topic
        topics = ['topic1','topic2']
        expected_output = {'topic1': [('token1', 0.8), ('token2', 0.6)],
                           'topic2' : [('token3', 0.4)]
                           }
        important_words_df = pd.DataFrame({'token': ['token1', 'token2', 'token3'],
                'attribution': [0.8, 0.6, 0.4],
                'topicId': [0, 0, 1]})
        
        class MyTopicExtractor:
            def searchTopics(self, topics,_):
                searchTopic =  topics[0]
                if searchTopic =="topic1":
                    return (_,_,_,0)
                if searchTopic =="topic2":
                    return (_,_,_,1)
        
        analyzer = ManualTopicAnalyzer(MyTopicExtractor())
        actual_output = analyzer.analyze(topics, important_words_df)
        self.assertEqual(expected_output, actual_output)

if __name__ == '__main__':
    unittest.main()