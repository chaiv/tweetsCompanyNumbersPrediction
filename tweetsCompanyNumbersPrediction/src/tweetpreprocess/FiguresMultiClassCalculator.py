'''
Created on 06.01.2025

@author: vital
'''
from tweetnumbersconnector.FinancialFiguresClassifier import FinancialFiguresMultiClassClassifier
from classifier.PredictionClassMappers import MULTICLASS_4

class FiguresMultiClassCalculator(object):



    def __init__(self, valueColumnName='percentChange',classColumnName='class'):
        self.valueColumnName = valueColumnName
        self.classColumnName = classColumnName

    def getFiguresWithClasses(self,figuresDf):
        predictionClassMapper = MULTICLASS_4 
        classes = [
            {"class_name":  predictionClassMapper.index_to_class(0), "from": -float('inf'), "to": -0.01},
            {"class_name": predictionClassMapper.index_to_class(1), "from": 0, "to": 15},
            {"class_name": predictionClassMapper.index_to_class(2), "from": 15, "to": 30},
            {"class_name": predictionClassMapper.index_to_class(3), "from": 30, "to": float('inf')}
        ]
        classifier = FinancialFiguresMultiClassClassifier(classes, percentChangeDfColumn='percent_change', classColumnName='class')
        classifier.add_classes(figuresDf)
        return figuresDf