'''
Created on 19.08.2026

Regression tests for the class boundary fixes.

The old FinancialFiguresMultiClassClassifier compared inclusively on both interval ends. With the
boundaries used by FiguresMultiClassCalculator (to=-0.01, from=0) this left percentage changes in
(-0.01, 0) without any class, and shared boundary values matched two classes at once. The intervals
are half-open [from, to) now. FiguresIncreaseDecreaseClassCalculator additionally makes explicit
whether its input contains ratios (1.02 for +2%) or percentages (2.0 for +2%): applying the ratio
threshold of 1.0 to percentages would label increases between 0% and 1% as decreases, which occurs
twice in the Tesla car sales data (2015Q3 +0.62%, 2018Q1 +0.37%).

@author: vital
'''
import unittest
import pandas as pd
from tweetpreprocess.FiguresMultiClassCalculator import FiguresMultiClassCalculator
from tweetpreprocess.FiguresIncreaseDecreaseClassCalculator import FiguresIncreaseDecreaseClassCalculator


class FiguresClassBoundariesTest(unittest.TestCase):

    def testNoGapBelowZero(self):
        df = pd.DataFrame({'percent_change': [-0.005, -0.0001, -0.01]})
        result = FiguresMultiClassCalculator().getFiguresWithClasses(df)
        # previously -0.005 and -0.0001 fell between to=-0.01 and from=0 and received None
        self.assertEqual([0, 0, 0], result['class'].tolist())

    def testBoundaryValuesBelongToExactlyOneClass(self):
        df = pd.DataFrame({'percent_change': [0.0, 15.0, 30.0, -5.0, 7.0, 20.0, 45.0]})
        result = FiguresMultiClassCalculator().getFiguresWithClasses(df)
        # half-open intervals: 0 -> class 1, 15 -> class 2, 30 -> class 3
        self.assertEqual([1, 2, 3, 0, 1, 2, 3], result['class'].tolist())

    def testRepresentativeValuesFollowDocumentedBoundaries(self):
        # Representative values away from the exact cut points retain the documented 0/15/30
        # classification.  The stale financial-CSV change_4 column is deliberately not the oracle.
        df = pd.DataFrame({'percent_change': [42.5, -22.5, 2.0, 9.4, 27.9, -17.5, 6.2, 10.4]})
        result = FiguresMultiClassCalculator().getFiguresWithClasses(df)
        self.assertEqual([3, 0, 1, 1, 2, 0, 1, 1], result['class'].tolist())

    def testBinaryCalculatorRatioMode(self):
        figures = pd.DataFrame({'percentChange': [1.12, 0.857, 1.0, 0.0]})
        result = FiguresIncreaseDecreaseClassCalculator().getFiguresWithClasses(figures)
        self.assertEqual([1, 0, 0, 0], result['class'].tolist())

    def testBinaryCalculatorPercentMode(self):
        # +0.62% and +0.37% are increases; the ratio threshold of 1.0 would call them decreases
        figures = pd.DataFrame({'percentChange': [0.62, 0.37, -0.5, 2.0, 0.0]})
        result = FiguresIncreaseDecreaseClassCalculator(valuesAreRatios=False).getFiguresWithClasses(figures)
        self.assertEqual([1, 1, 0, 1, 0], result['class'].tolist())


if __name__ == "__main__":
    unittest.main()
