'''
Created on 02.02.2022

@author: vital
'''

class FiguresPercentChangeCalculator(object):
    '''
    Adds percent changes of economic figures
    '''


    def __init__(self, percentChangeColumnName):
        self.percentChangeColumnName = percentChangeColumnName
        
        
    def getFiguresWithClasses(self,figuresDf):
        percentChanges = [] 
        previousFigureValue = None
        for i, row in figuresDf.iterrows():
            if(previousFigureValue != None):
                percentChanges.append(row[self.percentChangeColumnName]/previousFigureValue)
            else:
                percentChanges.append(None)    
            previousFigureValue = row[self.percentChangeColumnName]
        figuresWithPercentChangesDf = figuresDf.copy()     
        figuresWithPercentChangesDf["percentChange"]= percentChanges      
        return figuresWithPercentChangesDf[1:]