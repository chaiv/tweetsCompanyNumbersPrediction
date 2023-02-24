'''
Created on 24.02.2023

@author: vital
'''
import pandas as pd
import unittest
from unittest.mock import Mock
from pandas.util.testing import assert_frame_equal
from featureinterpretation.TweetTokenAttributionStore import TweetTokenAttributionStore

class TestTweetTokenAttributionStore(unittest.TestCase):
    def setUp(self):
        self.predictor = Mock()
        self.tweetTokenAttributionStore = TweetTokenAttributionStore(self.predictor)

    def test_add_from_df(self):
        # set up test data
        df = pd.DataFrame({
            "tweet_id": ["id1", "id2", "id3"],
            "body": ["this is the first tweet", "this is the second tweet", "this is the third tweet"]
        })

        # set up the mock predictor's response
        self.predictor.calculateWordScores.return_value = [
            (0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            (-0.1, -0.2, -0.3, -0.4, -0.5, -0.6),
            (0.5, 0.4, 0.3, 0.2, 0.1, 0.0)
        ]

        # call the method being tested
        self.tweetTokenAttributionStore.add_from_df(df)

        # check that the data was correctly added to the store
        expected_data = [
            ("id1", "this", 0.1, 0),
            ("id1", "is", 0.2, 1),
            ("id1", "the", 0.3, 2),
            ("id1", "first", 0.4, 3),
            ("id1", "tweet", 0.5, 4),
            ("id1", "<unk>", 0.6, 5),
            ("id2", "this", -0.1, 0),
            ("id2", "is", -0.2, 1),
            ("id2", "the", -0.3, 2),
            ("id2", "second", -0.4, 3),
            ("id2", "tweet", -0.5, 4),
            ("id2", "<unk>", -0.6, 5),
            ("id3", "this", 0.5, 0),
            ("id3", "is", 0.4, 1),
            ("id3", "the", 0.3, 2),
            ("id3", "third", 0.2, 3),
            ("id3", "tweet", 0.1, 4),
            ("id3", "<unk>", 0.0, 5),
        ]
        actual_data = self.tweetTokenAttributionStore.to_dataframe().to_records(index=False)
        self.assertListEqual(list(actual_data), expected_data)
        