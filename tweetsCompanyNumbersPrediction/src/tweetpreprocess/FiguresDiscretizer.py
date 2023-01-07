'''
Created on 01.02.2022

@author: vital
'''
from sklearn.preprocessing import KBinsDiscretizer

class FiguresDiscretizer(object):
    '''
    classdocs
    '''

    def __init__(self,figuresDf,classesNumber,valueColumnName='percentChange',classColumnName='class'):
        self.figuresDf = figuresDf
        self.valueColumnName = valueColumnName
        self.classColumnName = classColumnName
        self.discretizer = KBinsDiscretizer(n_bins=classesNumber, encode='ordinal', strategy='uniform')
        self.discretizer.fit(figuresDf[valueColumnName].values.reshape(-1, 1))
    

    
    def getFiguresWithClasses(self):
        classes = self.discretizer.transform(self.figuresDf[self.valueColumnName].values.reshape(-1, 1)).flatten()
        figuresDfWithClasses = self.figuresDf.copy()
        figuresDfWithClasses[self.classColumnName]=classes
        return figuresDfWithClasses