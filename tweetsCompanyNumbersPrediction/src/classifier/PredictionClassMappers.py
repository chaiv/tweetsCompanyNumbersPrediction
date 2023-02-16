'''
Created on 16.02.2023

@author: vital
'''
from classifier.PredictionClassMapper import PredictionClassMapper
BINARY_1_0 = PredictionClassMapper({0: 1, 1 : 0}, {1: "increase", 0: "decrease"})