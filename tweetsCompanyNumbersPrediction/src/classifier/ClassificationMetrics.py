'''
Created on 01.03.2023

@author: vital
'''
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

class ClassificationMetrics(object):

    def calculate_metrics(self, y_true,y_pred,pos_label,average='binary'):
        precision, recall, f1_score, support = precision_recall_fscore_support(
            y_true, y_pred, pos_label = pos_label, average=average
        )
        accuracy = accuracy_score(y_true,y_pred)
        return precision, recall, f1_score, support,accuracy 
    
    def classification_report(self, y_true, y_pred):
        report = classification_report(y_true, y_pred)
        return report