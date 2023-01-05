'''
Created on 22.01.2022

@author: vital
'''

class TweetNumbersConnector(object):
    '''
    Add tweets and economic figures together
    '''


    def __init__(self,fromTSPColumn = 'from_tsp',toTSPColumn ='to_tsp',valueColumn = 'value',postTSPColumn = 'post_tsp',postDateColumn = None):
        self.fromTSPColumn = fromTSPColumn
        self.toTSPColumn = toTSPColumn
        self.valueColumn = valueColumn
        self.postTSPColumn = postTSPColumn
        self.postDateColumn = postDateColumn
        
    def getFiguresValue(self,allNumbersDf,postTSP,postDate=None):
        value =  allNumbersDf.loc[
                (allNumbersDf[ self.fromTSPColumn] <= postTSP) 
                & 
                (allNumbersDf[ self.toTSPColumn] >= postTSP)
                ][self.valueColumn]
            
        if (len(value)==0):
            raise Exception("No value for tsp found: "+str(postTSP)+" "+str(postDate))
        
        if (len(value)>1):
            raise Exception("Only one value per tsp allowed, but multiple values match tsp: "+str(postTSP)+" "+str(postDate))
        return float(value)  
    
    
    def getTweetsWithNumbers(self,allTweetsDf, allNumbersDf):
        allTweetsWithNumbersDf =  allTweetsDf.copy()
        
        
        allTweetsWithNumbersDf[self.valueColumn] = allTweetsWithNumbersDf.apply(
            lambda x: self.getFiguresValue(
                allNumbersDf,
                x[self.postTSPColumn],
                postDate = (x[self.postDateColumn] if  (self.postDateColumn is not None) else None)  
                )
            ,axis = 1
        )
        return  allTweetsWithNumbersDf
    
