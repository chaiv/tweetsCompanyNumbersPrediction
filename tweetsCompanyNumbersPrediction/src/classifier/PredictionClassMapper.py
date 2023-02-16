'''
Created on 15.02.2023

@author: vital
'''
class PredictionClassMapper(object):
    '''
    classdocs
    '''


    def __init__(self,index_to_class_dict,class_to_economic_interpretation_dict):
        self.index_to_class_dict = index_to_class_dict
        self.class_to_economic_interpretation_dict = class_to_economic_interpretation_dict
    
    
    def class_to_index(self,classValue):
        for key, val in self.index_to_class_dict.items():
            if val == classValue:
                return key

    
    def index_to_class(self,index):
        return  self.index_to_class_dict[index]
    
    def class_to_economic_interpretation(self,classVal):   
        return self.class_to_economic_interpretation_dict[classVal]
        

    