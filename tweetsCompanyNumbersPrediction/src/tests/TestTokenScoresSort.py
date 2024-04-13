'''
Created on 11.04.2024

@author: vital
'''
import unittest
from featureinterpretation.TokenScoresSort import TokenScoresSort


class TestTokenScoresSort(unittest.TestCase):
    
    def setUp(self):
        self.sorter = TokenScoresSort()
        
    def test_sorted_tokens_and_scores_from_sublists_desc(self):
        tokens = [["apple", "banana"], ["cherry"]]
        scores = [[0.3,0.1 ], [0.2]]
        expected_tokens = ["apple", "cherry", "banana"]
        expected_scores = [0.3,0.2,0.1]
        sorted_tokens, sorted_scores = self.sorter.getSortedTokensAndScoresDescFromListOfLists(tokens, scores)
        self.assertEqual(sorted_tokens, expected_tokens)
        self.assertEqual(sorted_scores, expected_scores)
        
    def test_sorted_tokens_and_scores_desc(self):
        tokens = ["apple", "banana", "cherry"]
        scores = [3, 1, 2]
        expected_tokens = ["apple", "cherry", "banana"]
        expected_scores = [3,2,1]
        sorted_tokens, sorted_scores = self.sorter.getSortedTokensAndScoresDesc(tokens, scores)
        self.assertEqual(sorted_tokens, expected_tokens)
        self.assertEqual(sorted_scores, expected_scores)

    def test_sorted_tokens_and_scores(self):
        tokens = ["apple", "banana", "cherry"]
        scores = [3, 1, 2]
        expected_tokens = ["banana", "cherry", "apple"]
        expected_scores = [1, 2, 3]
        sorted_tokens, sorted_scores = self.sorter.getSortedTokensAndScoresAsc(tokens, scores)
        self.assertEqual(sorted_tokens, expected_tokens)
        self.assertEqual(sorted_scores, expected_scores)

    def test_sorted_tokens_and_scores_with_negative_scores(self):
        tokens = ["dog", "cat", "bird"]
        scores = [0, -2, 5]
        expected_tokens = ["cat", "dog", "bird"]
        expected_scores = [-2, 0, 5]
        sorted_tokens, sorted_scores = self.sorter.getSortedTokensAndScoresAsc(tokens, scores)
        self.assertEqual(sorted_tokens, expected_tokens)
        self.assertEqual(sorted_scores, expected_scores)

    def test_sorted_tokens_and_scores_same_scores(self):
        tokens = ["one", "two", "three"]
        scores = [7, 7, 7]
        expected_tokens = ["one", "two", "three"]
        expected_scores = [7, 7, 7]
        sorted_tokens, sorted_scores = self.sorter.getSortedTokensAndScoresAsc(tokens, scores)
        self.assertEqual(sorted_tokens, expected_tokens)
        self.assertEqual(sorted_scores, expected_scores)

    def test_input_length_mismatch_error(self):
        tokens = ["short"]
        scores = [1, 2]
        with self.assertRaises(ValueError):
            self.sorter.getSortedTokensAndScoresAsc(tokens, scores)

# To run the tests
if __name__ == '__main__':
    unittest.main()