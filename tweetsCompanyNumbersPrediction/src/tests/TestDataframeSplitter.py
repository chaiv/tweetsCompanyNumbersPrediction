'''
Created on 24.04.2023

@author: vital
'''
import unittest
import pandas as pd
from nlpvectors.DataframeSplitter import DataframeSplitter

class TestDataframeSplitter(unittest.TestCase):

    def setUp(self):
        self.splitter = DataframeSplitter()

    def test_splitDfByNSamplesForClass(self):
        data = {
            'col1': [1, 2, 3, 4, 5, 6, 7, 8,9],
            'class': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B','A']
        }
        df = pd.DataFrame(data)
        split_size = 2
        expected_num_splits = 5

        result = self.splitter.splitDfByNSamplesForClass(df, split_size)
        self.assertEqual(len(result), expected_num_splits)

        for split in result:
            # Check if all rows in the split belong to the same class
            unique_classes = split['class'].unique()
            self.assertEqual(len(unique_classes), 1)

            # Check if the split size is correct or smaller for remaining rows
            self.assertTrue(len(split) <= split_size)


if __name__ == "__main__":
    unittest.main()