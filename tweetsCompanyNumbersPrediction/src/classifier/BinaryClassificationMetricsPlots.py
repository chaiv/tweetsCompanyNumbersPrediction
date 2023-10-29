'''
Created on 29.10.2023

@author: vital
'''
import matplotlib.pyplot as plt
from classifier.BinaryClassificationMetrics import BinaryClassificationMetrics

class BinaryClassificationMetricsPlots(object):
    '''
    classdocs
    '''


    def __init__(self, metrics: BinaryClassificationMetrics):
        self.metrics = metrics
    
    
    def createRocAUCAndPrAucPlots(self,y_true,y_pred):
        plt.figure(figsize=(12, 5))
        
        fpr,tpr,roc_auc = BinaryClassificationMetrics().calculate_roc_auc(y_true, y_pred)
        
        # ROC Curve
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        
        precision, recall, pr_auc = BinaryClassificationMetrics().calculate_pr_auc(y_true, y_pred)
        
        # Precision-Recall Curve
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, color='darkorange', lw=2, label=f'AUC = {pr_auc:.2f}')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        
        # Show the plots
        plt.tight_layout()
        plt.show()
        