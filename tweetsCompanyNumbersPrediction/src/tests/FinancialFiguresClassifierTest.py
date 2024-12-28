'''
Created on 17.12.2024

@author: vital
'''
import unittest
import pandas as pd
from tweetnumbersconnector.FinancialFiguresClassifier import FinancialFiguresClassifier

class FinancialFiguresClassifierTest(unittest.TestCase):


    def test_change_classifier(self):    
        data = {
            'percent_change': [-5, 8, 15, 25]
        }
        df = pd.DataFrame(data)
    
        classes = [
            {"class_name": 0, "from": -float('inf'), "to": -0.01},
            {"class_name": 1, "from": 0, "to": 10},
            {"class_name": 2, "from": 10, "to": 20},
            {"class_name": 3, "from": 20, "to": float('inf')}
        ]
    
        classifier = FinancialFiguresClassifier(classes, percentChangeDfColumn='percent_change', classColumnName='multi_class')
    
        classifier.add_classes(df)
        counts = classifier.calculate_counts(df)

        expected_multi_class = [0, 1, 2, 3]
        expected_counts = {0: 1, 1: 1, 2: 1, 3: 1}
    
        assert df['multi_class'].tolist() == expected_multi_class, "Classification of multi_class failed."
        assert counts == expected_counts, "Counts calculation for multi_class failed."



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()