'''
Created on 12.03.2023

@author: vital
'''
import unittest
import pandas as pd
from classifier.TrainValTestSplitter import TrainValTestSplitter

class TestTrainValTestSplitter(unittest.TestCase):    
    def test_train_val_test_splitter(self):
        # create a small sample dataframe
        data = {
            'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'B': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'C': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        }
        df = pd.DataFrame(data)
        
        # instantiate the splitter
        splitter = TrainValTestSplitter(test_frac=0.3, val_frac=0.3)
        
        # get the split dataframes
        train_df, val_df, test_df = splitter.split(df)
        
        # check the number of rows in each dataframe
        self.assertEqual(len(train_df) + len(val_df) + len(test_df), len(df))
        self.assertEqual(len(train_df), 7)
        self.assertEqual(len(val_df), 3)
        self.assertEqual(len(test_df), 3)
        
        # check that the order of rows is maintained
        train_indices = list(train_df.index)
        val_indices = list(val_df.index)
        test_indices = list(test_df.index)
        all_indices = train_indices + val_indices + test_indices
        
        self.assertEqual(all_indices, list(df.index))
        self.assertEqual(sorted(train_indices), train_indices)
        self.assertEqual(sorted(val_indices), val_indices)
        self.assertEqual(sorted(test_indices), test_indices)