'''
Created on 24.04.2023

@author: vital
'''
import unittest
import pandas as pd
from nlpvectors.DataframeSplitter import DataframeSplitter
from sklearn.model_selection._split import KFold
from nlpvectors.VocabularyCreator import SEP_TOKEN

class TestDataframeSplitter(unittest.TestCase):

    def setUp(self):
        self.splitter = DataframeSplitter()
    
    def testClassCountsOfSplitsByIndexes(self):
        data = {
            "tweet_id" : [4,1,8,3,7],
            'class': [0, 1, 1, 0, 1]
        }
        df = pd.DataFrame(data)
        splits = self.splitter.getSplitIds(df, 2)
        splitIndexes = [1,2]
        resultCounts = self.splitter.getClassCountsOfSplitsByIndexes(df, splits,splitIndexes)
        self.assertEqual(resultCounts[0],0)
        self.assertEqual(resultCounts[1],2)
    
     
     
    def testClassCountsOfSplits(self):
        data = {
            "tweet_id" : [4,1,8,3,7],
            'class': [0, 1, 1, 0, 1]
        }
        df = pd.DataFrame(data)
        splits = self.splitter.getSplitIds(df, 2)
        resultCounts = self.splitter.getClassCountsOfSplits(df, splits)
        self.assertEqual(resultCounts[0],1)
        self.assertEqual(resultCounts[1],2)
        
         
     
    def testDfWithGroupedTweets(self):
        data = {
            "tweet_id" : [0,1,2,3,4],
            "body" : ["tweet_0","tweet_1","tweet_2","tweet_3","tweet_4"],
            'class': ['A', 'A', 'B', 'B', 'A']
        }
        df = pd.DataFrame(data)
        resultDf = self.splitter.getDfWithGroupedTweets(df,2)
        self.assertEqual(3,len(resultDf))
        self.assertEqual([0, 1],resultDf.iloc[0]["tweet_ids"])
        self.assertEqual([4],resultDf.iloc[1]["tweet_ids"])
        self.assertEqual([2, 3],resultDf.iloc[2]["tweet_ids"])
        self.assertEqual('tweet_0'+SEP_TOKEN+'tweet_1',resultDf.iloc[0]["body"]) 
        self.assertEqual('B',resultDf.iloc[2]["class"]) 
        
        
    def testFlattenFoldsAndUseAsIndexes(self):
        data = {
            "tweet_id" : [0,1,2,3,4],
            'class': ['A', 'A', 'B', 'B', 'A']
        }
        df = pd.DataFrame(data)
        splits = self.splitter.getSplitIds(df, 2)
        kfold = KFold(n_splits=2, shuffle=True, random_state=1337)
        folds = list(enumerate(kfold.split(splits)))
        fold_0, (train_idx_0, test_idx_0) = folds[0]
        fold_1, (train_idx_1, test_idx_1) = folds[1]
        train_rows_0 = self.splitter.getIdsOfSplitsAsFlattenedList(splits,train_idx_0)
        test_rows_0 = self.splitter.getIdsOfSplitsAsFlattenedList(splits,test_idx_0)
        train_rows_1 = self.splitter.getIdsOfSplitsAsFlattenedList(splits,train_idx_1)
        test_rows_1 = self.splitter.getIdsOfSplitsAsFlattenedList(splits,test_idx_1)
        self.assertEqual([4],train_rows_0)
        self.assertEqual([0,1,2,3],test_rows_0)
        self.assertEqual([0,1,2,3],train_rows_1)
        self.assertEqual([4],test_rows_1)
        self.assertEqual(4,len(df.iloc[test_rows_0]))
        
        
    def testKFoldWithSplits(self):
        data = {
            "tweet_id" : [0,1,2,3,4],
            'class': ['A', 'A', 'B', 'B', 'A']
        }
        df = pd.DataFrame(data)
        result = self.splitter.getSplitIds(df, 2)
        
        kfold = KFold(n_splits=2, shuffle=True, random_state=1337)
        folds = list(enumerate(kfold.split(result)))
        firstElementOfTrainIndicesOfFirstFold = folds[0][1][0].tolist()
        self.assertEqual( firstElementOfTrainIndicesOfFirstFold,[1])
        secondElementOfTrainIndicesOfFirstFold = folds[0][1][1].tolist()
        self.assertEqual(secondElementOfTrainIndicesOfFirstFold,[0,2])

     

    def test_splitDfByNSamplesForClass(self):
        data = {
            "tweet_id" : [0,1,2,3,4,5,6,7,8],
            'class': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B','A']
        }
        df = pd.DataFrame(data)
        split_size = 2
        expected_num_splits = 5

        result = self.splitter.getSplitIds(df, split_size)
        self.assertEqual(len(result), expected_num_splits)
        self.assertEqual([0,1],result[0])
        self.assertEqual([4,5],result[1])
        self.assertEqual([8],result[2])
        self.assertEqual([2,3],result[3])
        self.assertEqual([6,7],result[4])

    def test_splitDfBySingleSampleForClass(self):
        data = {
            "tweet_id" : [0,1,2,3],
            'class': ['A', 'A', 'B','A']
        }
        df = pd.DataFrame(data)
        split_size = 1
        expected_num_splits = 4

        result = self.splitter.getSplitIds(df, split_size)
        self.assertEqual(len(result), expected_num_splits)
        self.assertEqual([0],result[0])
        self.assertEqual([1],result[1])
        self.assertEqual([3],result[2])
        self.assertEqual([2],result[3])
  

if __name__ == "__main__":
    unittest.main()