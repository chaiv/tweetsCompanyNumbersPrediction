'''
Created on 01.03.2023

@author: vital
'''
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

class BinaryClassificationMetrics(object):

    def calculate_metrics(self, y_true,y_pred,pos_label,average='binary'):
        precision, recall, f1_score, support = precision_recall_fscore_support(
            y_true, y_pred, pos_label = pos_label, average=average
        )
        accuracy = accuracy_score(y_true,y_pred)
        return precision, recall, f1_score, support,accuracy 
    
    def calculate_mcc(self, y_true,y_pred):
        return matthews_corrcoef(y_true, y_pred)
   
    def calculate_roc_auc(self,y_true,y_pred):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        return fpr,tpr,roc_auc
    
    def calculate_pr_auc(self,y_true,y_pred):
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = average_precision_score(y_true,y_pred)
        return precision, recall, pr_auc
        
    
    def classification_report(self, y_true, y_pred):
        report = classification_report(y_true, y_pred)
        return report