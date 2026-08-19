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
        # Half-open intervals, so 0 belongs to class 1 and there is no uncovered range below 0.
        classes = [
            {"class_name":  predictionClassMapper.index_to_class(0), "from": -float('inf'), "to": 0},
            {"class_name": predictionClassMapper.index_to_class(1), "from": 0, "to": 15},
            {"class_name": predictionClassMapper.index_to_class(2), "from": 15, "to": 30},
            {"class_name": predictionClassMapper.index_to_class(3), "from": 30, "to": float('inf')}
        ]
        classifier = FinancialFiguresMultiClassClassifier(classes, percentChangeDfColumn='percent_change', classColumnName='class')
        classifier.add_classes(figuresDf)
        # Some archived financial CSVs carry a precomputed 'change_4' column produced with older
        # boundaries (roughly 0/10/25 instead of 0/15/30); it disagrees with the recomputed classes,
        # e.g. Amazon 2019Q1 (+27.9%: 3 vs 2) and 2019Q4 (+10.4%: 2 vs 1). The recomputed 'class'
        # column is authoritative; warn so the stale column is not mistaken for it.
        if 'change_4' in figuresDf.columns:
            comparable = figuresDf['percent_change'].notna()
            mismatches = int((figuresDf.loc[comparable, 'change_4'] != figuresDf.loc[comparable, 'class']).sum())
            if mismatches > 0:
                print("WARNING: precomputed column 'change_4' disagrees with the recomputed classes in %d rows "
                      "and is not used by the pipeline." % mismatches)
        return figuresDf