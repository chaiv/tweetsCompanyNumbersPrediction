'''
Created on 02.02.2022

@author: vital
'''

class FiguresPercentChangeCalculator(object):
    '''
    Adds percent changes of economic figures
    '''


    def __init__(self, valueColumnName='value',percentChangeColumnName='percentChange'):
        self.valueColumnName = valueColumnName
        self.percentChangeColumnName = percentChangeColumnName
        
        
    def getFiguresWithClasses(self,figuresDf):
        percentChanges = [] 
        previousFigureValue = None
        for i, row in figuresDf.iterrows():
            if(previousFigureValue != None):
                percentChanges.append(row[self.valueColumnName]/previousFigureValue)
            else:
                percentChanges.append(None)    
            previousFigureValue = row[self.valueColumnName]
        figuresWithPercentChangesDf = figuresDf.copy()     
        figuresWithPercentChangesDf[self.percentChangeColumnName]= percentChanges      
        return figuresWithPercentChangesDf[1:]