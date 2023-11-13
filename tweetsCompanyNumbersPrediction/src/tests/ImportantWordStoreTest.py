'''
Created on 17.02.2023

@author: vital
'''
import pandas as pd
import unittest
from featureinterpretation.ImportantWordsStore import ImportantWordStore,\
    createImportantWordStore, flatten_dict_lists, pad_dict_lists
from featureinterpretation.WordScoresWrapper import WordScoresWrapper
from nlpvectors.TweetGroup import TweetGroup



    
    


class TestImportantWordStore(unittest.TestCase):
    
  
    def testTransformDict(self):
        data = {"id":[0,1],"class":[2,3]}
        transformed_data = flatten_dict_lists(data)
        self.assertEqual([0,1],transformed_data["id"])
        self.assertEqual([2,3],transformed_data["class"])


    def testOneWordScoreWrapper(self):
        wordScoreWrapper = WordScoresWrapper(
        TweetGroup(sentences=["tweet 1", "second tweet 2"],sentenceIds=[0,1],totalTokenIndexes=[[1,2],[3,4,5]],
                                 totalTokens=[["tweet", "1"], ["second", "tweet","2"]],
                                 totalFeatureVector=[1,2,0,1,2,2],separatorIndexesInFeatureVector=[2],label = 0),
        [[0.1,0.2],[0.3,0.4,0.5]]
        )
        self.assertEquals(1.5,wordScoreWrapper.getAttributionsSum())
        wordscoreWrappers =[wordScoreWrapper ]
        predictions = [1]
        result = createImportantWordStore(wordscoreWrappers,predictions)
        df = result.to_dataframe()
        self.assertEqual(5,len(df))
        self.assertEqual(0,df["tweet_id"].iloc[0])
        self.assertEqual(1,df["token_index"].iloc[0])
        self.assertEqual("tweet",df["token"].iloc[0])
        self.assertEqual(0.1,df["token_attribution"].iloc[0])
        self.assertEqual(1,df["prediction"].iloc[0])
        self.assertEqual(1.5,df["tweet_attribution"].iloc[0])
   
           
    
