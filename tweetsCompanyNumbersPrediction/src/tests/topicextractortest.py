'''
Created on 29.01.2022

@author: vital
'''
import unittest
import pandas as pd
from topicmodelling.TopicExtractor import TopicExtractor
from topicmodelling.TopicModelCreator import TopicModelCreator
from tweetpreprocess.TweetDataframeQuery import TweetDataframeQuery
from tweetpreprocess.TweetQueryParams import TweetQueryParams
from tweetpreprocess.DataDirHelper import DataDirHelper


class TopicExtractorTest(unittest.TestCase):
    
    modelpath =  DataDirHelper().getDataDir()+ "companyTweets\TopicModelAAPLFirst1000"
    topicExtractor = TopicExtractor(TopicModelCreator().load(modelpath))
    
    def testWhenFindSentencesFromTopicThenOk(self):
        documents, document_scores, document_ids = self.topicExtractor.search_documents_by_topic(0, 5)
        sampleId = 551094968698171392
        self.assertEqual(
            sampleId,
                document_ids[0]
            )
        
        tweetsDf = pd.DataFrame(
                  [
                   (1111),    
                  (sampleId),
                   (1110)
                  ],
                  columns=['tweet_id']
                  )
        resultDf = TweetDataframeQuery().query(tweetsDf, TweetQueryParams(tweetIds=document_ids))
        self.assertEquals(1,resultDf.size)  
        self.assertEquals(sampleId,resultDf.iloc[0]['tweet_id'])  
        
    
    
    def testGetDocumentVector(self):
        self.assertAlmostEqual(0.01063707,self.topicExtractor.getDocumentVectorByTweetIndex(0)[0],4)  
        self.assertAlmostEqual(0.01063707,self.topicExtractor.getDocumentVectorByTweetId(550441509175443456)[0],4)  
   
    def testAllDocumentVectors(self):
        self.assertEquals(
            self.topicExtractor.getDocumentVectorByTweetIndex(0).all(),
            self.topicExtractor.getDocumentVectorsAsArray()[0].all()
            )  
        pass
    
    def testGetDocumentVectorSize(self):
        self.assertEquals(300,self.topicExtractor.getFeatureVectorSize())  
        pass


    def testTopicExtraction(self):
        self.assertTrue(self.topicExtractor.getNumTopics()>0)
        topic_words, word_scores, topic_scores, topic_nums =  self.topicExtractor.searchTopics(keywords=["work"], num_topics=self.topicExtractor.getNumTopics())
        self.assertAlmostEqual(topic_scores[0],0.9831678519450329,3 )
        pass
    
    def testWordIndexes(self):
       self.assertEquals(2706, len(self.topicExtractor.getWordIndexes()))
       
    def testWordVectors(self):
        self.assertEquals(300,len(self.topicExtractor.getWordVectorsOfWords(["apple","interesting"])[0]))
        self.assertEquals(2,len(self.topicExtractor.getWordVectorsOfWords(["apple","interesting"])))
        
    
    
    
        

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'TopicExtractorTest.testTopicExtraction']
    unittest.main()
