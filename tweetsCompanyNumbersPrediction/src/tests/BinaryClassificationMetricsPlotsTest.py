'''
Created on 29.10.2023

@author: vital
'''
import unittest
from classifier.BinaryClassificationMetricsPlots import BinaryClassificationMetricsPlots
from classifier.BinaryClassificationMetrics import BinaryClassificationMetrics


class BinaryClassificationMetricsPlotsTest(unittest.TestCase):


    def testRocAUCAndPRAUCPlots(self):
        y_true = [0, 1, 0, 1, 1, 0]
        y_pred = [0, 0, 0, 1, 1, 1]
        BinaryClassificationMetricsPlots(BinaryClassificationMetrics()).createRocAUCAndPrAucPlots(y_true, y_pred)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()