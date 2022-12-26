'''
Created on 26.12.2022

@author: vital
'''
import datetime

class DateToTSP(object):

    def __init__(self, dateFormat='%d/%m/%Y'):
        self.dateFormat = dateFormat
   
    def dateStrToTSPInt(self,dateStr):
        date=datetime.datetime.strptime(dateStr,self.dateFormat)
        return int(round( date.timestamp()))
        