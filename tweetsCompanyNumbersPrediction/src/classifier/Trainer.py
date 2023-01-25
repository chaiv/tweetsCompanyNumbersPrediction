'''
Created on 25.01.2023

@author: vital
'''
import torch
class Trainer(object):

    def __init__(self,loss_function,optimizer, epochs = 10,deviceToUse = torch.device("cuda:0") ):
        self.epochs = epochs
        self.deviceToUse = deviceToUse
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.losses = []
        self.accur = []
    
    def train(self,model,x_data,y_data,trainloader):
        for i in range(self.epochs):
            for j,(x_train,y_train) in enumerate(trainloader):
    
                #calculate output
                output = model(x_train)
 
                #calculate loss
                loss =  self.loss_function(output,y_train.reshape(-1,1))
 
                #accuracy
                predicted = model(torch.tensor(x_data,dtype=torch.float32).to(self.deviceToUse ))
                acc = (predicted.reshape(-1).detach().cpu().numpy().round() == y_data).mean()    #backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.losses.append(loss)
                self.accur.append(acc)
                print("epoch {}\tloss : {}\t accuracy : {}".format(i,loss,acc))    