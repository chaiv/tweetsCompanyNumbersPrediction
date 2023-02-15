'''
Created on 15.02.2023

@author: vital
'''

class PredictionClassMapper(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        pass
    
    
    def index_to_class(self,index):
        pass 
    
    def class_to_economic_interpretation(self,classVal):   
        pass
        

class BinaryClassMapper(PredictionClassMapper):
    
    def __init__(self):
        self.index_to_class_dict = {0: 1, 1 : 0}
        self.class_to_economic_interpretation_dict = {1: "increase", 0: "decrease"}
        pass
    
    
    def index_to_class(self,index):
        return  self.index_to_class_dict[index]
    
    def class_to_economic_interpretation(self,classVal):  
        return self.class_to_economic_interpretation_dict[classVal]
    