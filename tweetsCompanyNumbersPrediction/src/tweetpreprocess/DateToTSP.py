'''
Created on 26.12.2022

@author: vital
'''
import datetime

class DateTSPConverter(object):

    def __init__(self, dateFormat='%d/%m/%Y'):
        self.dateFormat = dateFormat
   
    def dateStrToTSPInt(self,dateStr):
        date=datetime.datetime.strptime(dateStr,self.dateFormat)
        return int(round( date.timestamp()))
    
    def tspIntToDate(self,tspInt):  
        return datetime.datetime.fromtimestamp(tspInt) 