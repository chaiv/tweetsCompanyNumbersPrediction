'''
Created on 20.04.2023

@author: vital
'''
import pandas as pd
import unittest
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

class TestKFold(unittest.TestCase):
    def setUp(self):
        # Define your dataframe
        self.df = pd.DataFrame({
            "text": ["text1", "text2", "text3"],
            "label": [0, 1, 0]
        })
        self.kfold = KFold(n_splits=3, shuffle=True, random_state=1337)

    def test_kfold_split(self):
        for fold, (train_idx, test_idx) in enumerate(self.kfold.split(self.df)):
            # Split the train set into train and validation sets
            train_idx, val_idx = train_test_split(train_idx, random_state=1337, test_size=0.3)
            train_df = self.df.iloc[train_idx]
            val_df = self.df.iloc[val_idx]
            test_df = self.df.iloc[test_idx]
            
            # Assert that the number of samples in the train, validation, and test sets sum up to the original number of samples
            self.assertEqual(len(train_df) + len(val_df) + len(test_df), len(self.df))
            
            # Assert that the labels in the train, validation, and test sets are balanced
            train_labels = train_df["label"].tolist()
            val_labels = val_df["label"].tolist()
            test_labels = test_df["label"].tolist()
            self.assertAlmostEqual(train_labels.count(0), train_labels.count(1), delta=1)
            self.assertAlmostEqual(val_labels.count(0), val_labels.count(1), delta=1)
            self.assertAlmostEqual(test_labels.count(0), test_labels.count(1), delta=1)