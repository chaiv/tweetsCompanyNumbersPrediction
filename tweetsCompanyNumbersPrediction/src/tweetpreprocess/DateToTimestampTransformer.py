'''
Created on 25.12.2022

@author: vital
'''
import datetime

class DateToTimestampDataframeTransformer(object):


    def __init__(self, fromDateColumn='from_date',toDateColumn='to_date',fromTSPColumn = 'from_tsp',toTSPColumn = 'to_tsp',dateFormat='%d/%m/%Y'):
        self.fromDateColumn = fromDateColumn
        self.toDateColumn = toDateColumn
        self.fromTSPColumn = fromTSPColumn
        self.toTSPColumn = toTSPColumn
        self.dateFormat = dateFormat
        
        
    def dateToTSPInt(self,dateStr):
        date=datetime.datetime.strptime(dateStr,self.dateFormat)
        return int(round( date.timestamp()))
        
    def addTSPColumn(self,numbersDf,dateColumn,tspColumn):    
        numbersDf[tspColumn] = numbersDf.apply(lambda row: self.dateToTSPInt(row[dateColumn]), axis=1)
        return numbersDf        
        
    def addTimestampColumns(self,numbersDf):    
        numbersDf = self.addTSPColumn(numbersDf,self.fromDateColumn,  self.fromTSPColumn)
        numbersDf = self.addTSPColumn(numbersDf,self.toDateColumn,  self.toTSPColumn)
        return numbersDf
    
