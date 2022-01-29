'''
Created on 29.01.2022

@author: vital
'''
import unittest
import pandas as pd
from topicmodelling.TopicExtractor import TopicExtractor
from topicmodelling.TopicModelCreator import TopicModelCreator

class TopicExtractorTest(unittest.TestCase):


    def testTopicExtraction(self):
        tweets = pd.DataFrame(
                  [
                  ("Images showing the first Battery Swap Station http://imgur.com/dummylink via @imgur"),
                  ("company next charger will automatically connect to your car http://flip.it/dummylink"),
                  ("Bank Of Dummy Just Released A New Auto Report (And They Discussed companies): Bank of Dummy... http://sgfr.us/dummylink"),
                  ("Electric Vehicle Battery Energy Density - Accelerating the Development Timeline.  https://lnkd.in/dummylink"),
                  ("Max Mustermann Interview: World Needs Hundreds of Gigafactories. https://lnkd.in/dummylink")
                  ],
                  columns=["body"]
                  )
        topicExtractor = TopicExtractor(TopicModelCreator().createModel(tweets["body"].tolist()))
        print(topicExtractor.getNumTopics())
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'TopicExtractorTest.testTopicExtraction']
    unittest.main()