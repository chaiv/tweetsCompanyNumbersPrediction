'''
Created on 22.01.2022

@author: vital
'''
from tweetpreprocess.DateToTSP import DateTSPConverter

class TweetNumbersConnector(object):
    '''
    Connects tweets and economic figures
    '''


    def __init__(self,fromTSPColumn = 'from_tsp',toTSPColumn ='to_tsp',valueColumn = 'value',postTSPColumn = 'post_tsp',dateTSPConverter = DateTSPConverter()
                 ):
        self.fromTSPColumn = fromTSPColumn
        self.toTSPColumn = toTSPColumn
        self.valueColumn = valueColumn
        self.postTSPColumn = postTSPColumn
        self.dateTSPConverter = dateTSPConverter
        
    def getFiguresValue(self,allNumbersDf,postTSP):
        value =  allNumbersDf.loc[
                (allNumbersDf[ self.fromTSPColumn] <= postTSP) 
                & 
                (allNumbersDf[ self.toTSPColumn] >= postTSP)
                ][self.valueColumn]
            
        if (len(value)==0):
            raise Exception("No numbers value for tsp found: "+str(postTSP)+" "+str(self.dateTSPConverter.tspIntToDate(postTSP)))
        
        if (len(value)>1):
            raise Exception("Only one numbers value per tsp allowed, but multiple values match tsp: "+str(postTSP)+" "+str(self.dateTSPConverter.tspIntToDate(postTSP)))
        return float(value)  
    
    
    def getTweetsWithNumbers(self,allTweetsDf, allNumbersDf):
        allTweetsWithNumbersDf =  allTweetsDf.copy()
        
        
        allTweetsWithNumbersDf[self.valueColumn] = allTweetsWithNumbersDf.apply(
            lambda x: self.getFiguresValue(
                allNumbersDf,
                x[self.postTSPColumn]
                )
            ,axis = 1
        )
        return  allTweetsWithNumbersDf
    
