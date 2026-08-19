'''
Created on 07.01.2023

@author: vital
'''

class FiguresIncreaseDecreaseClassCalculator(object):
    '''
    Adds classes 0 and 1 to changes of figures.

    valuesAreRatios states the semantics of the value column. FiguresPercentChangeCalculator writes
    RATIOS (current/previous, 1.02 for +2%) into a column named percentChange, and for ratios the
    increase threshold is 1.0. The archived financial CSVs however store true PERCENTAGES (2.0 for
    +2%) in percent_change, where the threshold has to be 0. Applying the ratio threshold to a
    percentage column silently labels increases between 0% and 1% as decreases; the Tesla car sales
    data contains two such quarters (2015Q3 +0.62%, 2018Q1 +0.37%).
    '''


    def __init__(self, valueColumnName='percentChange',classColumnName='class',valuesAreRatios=True):
        self.valueColumnName = valueColumnName
        self.classColumnName = classColumnName
        self.increaseThreshold = 1.0 if valuesAreRatios else 0.0

    def getFiguresWithClasses(self,figuresDf):
        figuresDfWithClasses = figuresDf.copy()
        figuresDfWithClasses[self.classColumnName] = figuresDfWithClasses.apply(
            lambda x: 1 if x[self.valueColumnName] > self.increaseThreshold else 0,
            axis=1
        )
        return figuresDfWithClasses
