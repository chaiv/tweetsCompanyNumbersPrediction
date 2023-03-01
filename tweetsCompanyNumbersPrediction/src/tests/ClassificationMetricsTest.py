'''
Created on 01.03.2023

@author: vital
'''
import unittest
from classifier.ClassificationMetrics import ClassificationMetrics


class ClassificationMetricsTest(unittest.TestCase):


    def test_calculate_metrics(self):
        y_true = [0, 1, 0, 1, 1, 0]
        y_pred = [0, 0, 0, 1, 1, 1]
        metrics = ClassificationMetrics()
        pos_label = 1
        precision, recall, f1_score, support, accuracy = metrics.calculate_metrics(y_true, y_pred, pos_label)
    
    # Assert the calculated values
        self.assertAlmostEqual(precision, 0.6666666666666666, places=5)
        self.assertAlmostEqual(recall, 0.6666666666666666, places=5)
        self.assertAlmostEqual(f1_score, 0.6666666666666666, places=5)
        self.assertIsNone(support)
        self.assertAlmostEqual(accuracy, 0.6666666666666666, places=5)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'ClassificationMetricsTest.testName']
    unittest.main()