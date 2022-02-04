'''
Created on 01.02.2022

@author: vital
'''
from sklearn.preprocessing import KBinsDiscretizer

class FiguresDiscretizer(object):
    '''
    classdocs
    '''

    def __init__(self,figuresDf,percentChangeColumnName, classesNumber):
        self.classesNumber = classesNumber
        self.figuresDf = figuresDf
        self.percentChangeColumnName = percentChangeColumnName
        self.discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
        self.discretizer.fit(figuresDf[percentChangeColumnName].values.reshape(-1, 1))
    

    
    def getFiguresWithClasses(self):
        classes = self.discretizer.transform(self.figuresDf[self.percentChangeColumnName].values.reshape(-1, 1)).flatten()
        figuresDfWithClasses = self.figuresDf.copy()
        figuresDfWithClasses["class"]=classes
        return figuresDfWithClasses