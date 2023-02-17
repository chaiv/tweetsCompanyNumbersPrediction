'''
Created on 16.02.2023

@author: vital
'''
from classifier.PredictionClassMapper import PredictionClassMapper
BINARY_0_1 = PredictionClassMapper({0: 0, 1 : 1}, { 0: "decrease",1: "increase"})