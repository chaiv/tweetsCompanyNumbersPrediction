'''
Created on 17.02.2023

@author: vital
'''
import unittest
from featureinterpretation.TokenAttributionStore import TokenAttributionStore

class TestTokenAttributionStore(unittest.TestCase):
    
    def test_to_dataframe(self):
        # create an instance of the class
        store = TokenAttributionStore()

        # add some data to the store
        id_value = 1
        tokens = ["the", "cat", "is", "on", "the", "mat"]
        attributions = [0.1, -0.2, 0.3, 0.05, -0.1, 0.15]
        store.add_data(id_value, tokens, attributions)

        id_value = 2
        tokens = ["the", "dog", "is", "in", "the", "yard"]
        attributions = [-0.05, 0.15, 0.2, 0.1, -0.1, 0.1]
        store.add_data(id_value, tokens, attributions)

        # convert the stored data to a dataframe
        df = store.to_dataframe()

        # check that the dataframe has the correct columns
        self.assertEqual(list(df.columns), ['id', 'token', 'attribution'])

        # check that the dataframe has the correct values
        expected_data = [
            (1, "the", 0.1),
            (1, "cat", -0.2),
            (1, "is", 0.3),
            (1, "on", 0.05),
            (1, "the", -0.1),
            (1, "mat", 0.15),
            (2, "the", -0.05),
            (2, "dog", 0.15),
            (2, "is", 0.2),
            (2, "in", 0.1),
            (2, "the", -0.1),
            (2, "yard", 0.1)
        ]
        for row, expected_row in zip(df.itertuples(index=False), expected_data):
            self.assertEqual(row, expected_row)