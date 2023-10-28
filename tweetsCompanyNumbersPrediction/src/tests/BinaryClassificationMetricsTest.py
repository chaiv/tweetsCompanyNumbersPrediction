'''
Created on 01.03.2023

@author: vital
'''
import unittest
from classifier.BinaryClassificationMetrics import BinaryClassificationMetrics


class BinaryClassificationMetricsTest(unittest.TestCase):
    
    def test_roc_auc(self):
        y_true = [0, 1, 0, 1, 1, 0]
        y_pred = [0, 0, 0, 1, 1, 1]
        fpr,tpr,roc_auc = BinaryClassificationMetrics().calculate_roc_auc(y_true, y_pred)
        self.assertAlmostEqual(roc_auc, 0.6666666666666667, places=5)
    
    
    def test_pr_auc(self):
        y_true = [0, 1, 0, 1, 1, 0]
        y_pred = [0, 0, 0, 1, 1, 1]
        precision, recall, pr_auc = BinaryClassificationMetrics().calculate_pr_auc(y_true, y_pred)
        self.assertAlmostEqual(pr_auc, 0.6111111111111112, places=5)

    def test_calculate_mcc(self):
        y_true = [0, 1, 0, 1, 1, 0]
        y_pred = [0, 0, 0, 1, 1, 1]
        mcc = BinaryClassificationMetrics().calculate_mcc(y_true, y_pred)
        self.assertAlmostEqual(mcc, 0.3333333333333333, places=5)


    def test_calculate_metrics(self):
        y_true = [0, 1, 0, 1, 1, 0]
        y_pred = [0, 0, 0, 1, 1, 1]
        metrics = BinaryClassificationMetrics()
        pos_label = 1
        precision, recall, f1_score, support, accuracy = metrics.calculate_metrics(y_true, y_pred, pos_label)

        self.assertAlmostEqual(precision, 0.6666666666666666, places=5)
        self.assertAlmostEqual(recall, 0.6666666666666666, places=5)
        self.assertAlmostEqual(f1_score, 0.6666666666666666, places=5)
        self.assertIsNone(support)
        self.assertAlmostEqual(accuracy, 0.6666666666666666, places=5)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'BinaryClassificationMetricsTest.testName']
    unittest.main()