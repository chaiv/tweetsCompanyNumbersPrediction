'''
Created on 17.01.2023

@author: vital
'''
import torch
x = torch.rand(5, 3)
print(x)
print(torch.cuda.is_available())
print(torch.zeros(1).cuda())
