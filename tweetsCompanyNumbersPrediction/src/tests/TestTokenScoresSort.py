'''
Created on 11.04.2024

@author: vital
'''
import unittest
from featureinterpretation.TokenScoresSort import TokenScoresSort


class TestTokenScoresSort(unittest.TestCase):
    

    def test_sorting_basic(self):
        sorter = TokenScoresSort()
        tokens = ["apple", "banana", "cherry"]
        scores = [3, 1, 2]
        expected = [("banana", 1), ("cherry", 2), ("apple", 3)]
        self.assertEqual(sorter.getSortedTokensWithScoresAsc(tokens, scores), expected)

    def test_sorting_with_negative_scores(self):
        sorter = TokenScoresSort()
        tokens = ["dog", "cat", "bird"]
        scores = [0, -2, 5]
        expected = [("cat", -2), ("dog", 0), ("bird", 5)]
        self.assertEqual(sorter.getSortedTokensWithScoresAsc(tokens, scores), expected)

    def test_sorting_with_same_scores(self):
        sorter = TokenScoresSort()
        tokens = ["one", "two", "three"]
        scores = [7, 7, 7]
        expected = [("one", 7), ("two", 7), ("three", 7)]  # Stability of sort depends on sort implementation
        self.assertEqual(sorter.getSortedTokensWithScoresAsc(tokens, scores), expected)

    def test_invalid_input_length_mismatch(self):
        sorter = TokenScoresSort()
        tokens = ["short"]
        scores = [1, 2]
        with self.assertRaises(ValueError):
            sorter.getSortedTokensWithScoresAsc(tokens, scores)

# To run the tests
if __name__ == '__main__':
    unittest.main()