'''
Created on 17.12.2024

@author: vital
'''
import pandas as pd
from PredictionModelPath import AMAZON_REVENUE_10, TESLA_CAR_SALES_10,\
     APPLE__EPS_10
from tweetnumbersconnector.FinancialFiguresClassifier import FinancialFiguresMultiClassClassifier



amazonRevenueDf = pd.read_csv(AMAZON_REVENUE_10.getFinancialNumbersPath())
teslaCarSalesDf = pd.read_csv(TESLA_CAR_SALES_10.getFinancialNumbersPath())
appleEpsDf = pd.read_csv(APPLE__EPS_10.getFinancialNumbersPath())
classes = [
            {"class_name": 0, "from": -float('inf'), "to": -0.01},
            {"class_name": 1, "from": 0, "to": 15},
            {"class_name": 2, "from": 15, "to": 30},
            {"class_name": 3, "from": 30, "to": float('inf')}
        ]

classifier = FinancialFiguresMultiClassClassifier(classes, percentChangeDfColumn='percent_change', classColumnName='multi_class')

classifier.add_classes(amazonRevenueDf)
classifier.add_classes(teslaCarSalesDf)
classifier.add_classes(appleEpsDf)
print(classifier.calculate_counts(amazonRevenueDf))
print(classifier.calculate_counts(teslaCarSalesDf))
print(classifier.calculate_counts(appleEpsDf))
