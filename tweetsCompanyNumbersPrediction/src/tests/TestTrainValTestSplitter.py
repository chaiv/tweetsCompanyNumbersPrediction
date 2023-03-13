'''
Created on 12.03.2023

@author: vital
'''
import unittest
import pandas as pd
from classifier.TrainValTestSplitter import TrainValTestSplitter

class TestTrainValTestSplitter(unittest.TestCase):    
    def test_split_method(self):
        # Create a sample input DataFrame
        input_df = pd.DataFrame({'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                 'col2': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']})



        # Expected output
        expected_train_df = pd.DataFrame({'col1': [1, 2, 3, 4, 5, 6],
                                           'col2': ['A', 'B', 'C', 'D', 'E', 'F']})
        expected_val_df = pd.DataFrame({'col1': [7, 8],
                                         'col2': ['G', 'H']},index=[6, 7])
        expected_test_df = pd.DataFrame({'col1': [9, 10],
                                          'col2': ['I', 'J']},index=[8, 9])
        
        train_df, val_df, test_df = TrainValTestSplitter().split(input_df)

        # Check if the output matches the expected output
        self.assertTrue( train_df.equals(expected_train_df))
        self.assertTrue(val_df.equals(expected_val_df))
        self.assertTrue(test_df.equals(expected_test_df))
        