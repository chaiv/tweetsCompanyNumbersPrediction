'''
Created on 25.12.2022

@author: vital
'''
from tweetpreprocess.DateToTSP import DateTSPConverter

class DateToTimestampDataframeTransformer(object):


    def __init__(self, dateColumnNames=['from_date','to_date'],tspColumnNames = ['from_tsp','to_tsp'],dateToTSP=DateTSPConverter()):
        if len(dateColumnNames) != len(tspColumnNames):
            raise Exception("Date and tsp column number must match!")
        self.dateToTSP = dateToTSP
        self.dateColumnNames = dateColumnNames
        self.tspColumnNames = tspColumnNames
        
        
    def addTSPColumn(self,numbersDf,dateColumn,tspColumn):    
        numbersDf[tspColumn] = numbersDf.apply(lambda row: self.dateToTSP.dateStrToTSPInt(row[dateColumn]), axis=1)
        return numbersDf        
        
    def addTimestampColumns(self,numbersDf): 
        for i in range(len(self.dateColumnNames)):   
            numbersDf = self.addTSPColumn(numbersDf,self.dateColumnNames[i], self.tspColumnNames[i])
        return numbersDf
    
