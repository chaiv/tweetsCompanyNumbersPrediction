'''
Created on 11.11.2023

@author: vital
'''
import unittest
from nlpvectors.TweetGroup import split_list_on_indices

SEPARATOR_ID = 2

class TestSplitLists(unittest.TestCase):
    

        
    def testSplitLists(self):
        l = [1, 1, SEPARATOR_ID, 1, 1, SEPARATOR_ID, 1]
        self.assertEqual([[1, 1], [1, 1], [1]], split_list_on_indices(l, [2, 5]))
    
    def testSplitListsSingleEnd(self):
        l = [1, 1, SEPARATOR_ID]
        self.assertEqual([[1, 1]], split_list_on_indices(l, [2]))
    
    def testSplitListsSingleBeginn(self):
        l = [SEPARATOR_ID, 1, 1]
        self.assertEqual([[1, 1]], split_list_on_indices(l, [0]))
    
    def testSplitListsSingleMiddle(self):
        l = [1, SEPARATOR_ID, 1]
        self.assertEqual([[1], [1]], split_list_on_indices(l, [1]))
    
    def testSplitListsDoubleMiddle(self):
        l = [1, SEPARATOR_ID, SEPARATOR_ID, 1]
        self.assertEqual([[1], [1]], split_list_on_indices(l, [1, 2]))
    
    def testSplitListsDoubleBeginn(self):
        l = [SEPARATOR_ID, SEPARATOR_ID, 1, 1]
        self.assertEqual([[1, 1]], split_list_on_indices(l, [0, 1]))
    
    def testSplitListsDoubleEnd(self):
        l = [1, 1, SEPARATOR_ID, SEPARATOR_ID]
        self.assertEqual([[1, 1]], split_list_on_indices(l, [2, 3]))
    
    def testSplitListsEmptyIndices(self):
        l = [1, 1]
        self.assertEqual([l], split_list_on_indices(l, []))
    
    def testSplitListsNotExistingIndices(self):
        l = [1, 1]
        self.assertEqual([[1, 1]], split_list_on_indices(l, [2]))
    
    def testSplitListsCombination(self):
        l = [SEPARATOR_ID, 1, SEPARATOR_ID, SEPARATOR_ID, 1, 1, SEPARATOR_ID, SEPARATOR_ID]
        self.assertEqual([[1], [1, 1]], split_list_on_indices(l, [0, 2, 3, 6, 7]))



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()