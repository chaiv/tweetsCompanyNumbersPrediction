'''
Created on 07.01.2023

@author: vital
'''

class FiguresIncreaseDecreaseClassCalculator(object):
    '''
    Adds classes 0 und 1 to percent changes of figures
    '''


    def __init__(self, valueColumnName='percentChange',classColumnName='class'):
        self.valueColumnName = valueColumnName
        self.classColumnName = classColumnName

    def getFiguresWithClasses(self,figuresDf):
        figuresDfWithClasses = figuresDf.copy()
        figuresDfWithClasses[self.classColumnName]= figuresDfWithClasses.apply(
            lambda x: 
            (1 if  (x[self.valueColumnName]>1.0) else 0),
            axis = 1
        ); 
        return figuresDfWithClasses    