'''
Created on 25.01.2023

@author: vital
'''
import torch
class Trainer(object):

    def __init__(self,loss_function,optimizer, model, epochs = 10,deviceToUse = torch.device("cuda:0") ):
        self.epochs = epochs
        self.deviceToUse = deviceToUse
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.model = model
        self.losses = []
        self.accur = []
        
        
    def test(self):
        pass
    
    
    def calculateAccuracy(self,x,y):
        predicted = self.model(torch.tensor(x,dtype=torch.float32).to(self.deviceToUse ))
        return (predicted.reshape(-1).detach().cpu().numpy().round() == y).mean() 
    
    def calculateLoss(self, y, predicted):
        return self.loss_function(predicted, y.reshape(-1, 1))
    
    def train(self,x_test,y_test,trainloader):
        for i in range(self.epochs):
            for j,(x_train,y_train) in enumerate(trainloader):
    
                #calculate output
                output = self.model(x_train)
 
                #calculate loss
                loss = self.calculateLoss(y_train.reshape(-1,1),output)
 
                #accuracy
                acc = self.calculateAccuracy(x_test,y_test)
                #backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.losses.append(loss)
                self.accur.append(acc)
                print("epoch {}\tloss : {}\t accuracy : {}".format(i,loss,acc))    