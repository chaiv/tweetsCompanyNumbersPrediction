'''
Created on 16.02.2023

@author: vital
'''
from classifier.PredictionClassMapper import PredictionClassMapper
BINARY_0_1 = PredictionClassMapper({0: 0, 1 : 1}, { 0: "decrease",1: "increase"})
MULTICLASS_4 = PredictionClassMapper({0: 0, 1 : 1, 2: 2, 3: 3}, { 0: "decrease",1: "weak increase",2: "moderate increase",3: "strong increase"})
