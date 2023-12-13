# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 01:04:24 2023

@author: Omar Al-maqtari
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


class DiceLoss(nn.Module):
    def __init__(self, threshold):
        super(DiceLoss, self).__init__()
        self.threshold = threshold

    def forward(self, SR, GT, smooth=1e-8): 
        
        SR = SR.view(-1)
        GT = GT.view(-1)
        Inter = torch.sum((SR>self.threshold)&(GT>0.8))
        Union = torch.sum(SR>self.threshold) + torch.sum(GT>0.8)
        Dice = float(2.*Inter)/(float(Union) + smooth)
        
        return 1 - Dice

class IoULoss(nn.Module):
    def __init__(self, threshold):
        super(IoULoss, self).__init__()
        self.threshold = threshold

    def forward(self, SR, GT, smooth=1e-8):
        SR = SR.view(-1)
        GT = GT.view(-1) 
        Inter = torch.sum((SR>self.threshold)&(GT>0.8))
        Union = torch.sum(SR>self.threshold) + torch.sum(GT>0.8) - Inter
        IoU = float(Inter)/(float(Union) + smooth)
                
        return 1 - IoU

class mIoULoss(nn.Module):
    def __init__(self, threshold):
        super(mIoULoss, self).__init__()
        self.threshold = threshold
    
    def forward(self, SR, GT, smooth=1e-8):
        SR = SR.view(-1)
        GT = GT.view(-1)
        
        # IoU of Foreground
        Inter1 = torch.sum((SR>self.threshold)&(GT>0.8))
        Union1 = torch.sum(SR>self.threshold) + torch.sum(GT>0.8) - Inter1
        IoU1 = float(Inter1)/(float(Union1) + smooth)

        # IoU of Background
        Inter2 = torch.sum((SR<self.threshold)&(GT<0.8))
        Union2 = torch.sum(SR<self.threshold) + torch.sum(GT<0.8) - Inter2
        IoU2 = float(Inter2)/(float(Union2) + smooth)

        mIoU = (IoU1 + IoU2) / 2
                
        return 1 - mIoU

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, size_average=True):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.size_average = size_average
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.criterion(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.size_average:
            return F_loss.mean()
        else:
            return F_loss.sum()


def PRC(PC, RC, result_path, report_name):
    RC1 = []
    PC1 = []
    RC = list(map(list, zip(*RC)))
    PC = list(map(list, zip(*PC)))
    
    for i in range(len(RC)):
        RC1.append(np.sum(RC[i])/len(RC[i]))
        
    for i in range(len(PC)):
        PC1.append(np.sum(PC[i])/len(PC[i]))
    
    PC = np.fliplr([PC1])[0]  #to avoid getting negative AUC
    RC = np.fliplr([RC1])[0]  #to avoid getting negative AUC
    AUC_PC_RC = np.trapz(PC,RC)
    print("\nArea under Precision-Recall curve: " +str(AUC_PC_RC))
    plt.figure()
    plt.plot(RC,PC,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_PC_RC)
    plt.title('Precision - Recall curve')
    plt.legend(loc="lower right")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.savefig(result_path+report_name+'_Precision_recall.png')
    
    return RC, PC

def displayfigures(results, result_path, report_name):
    for i in range(len(results)):
        plt.Figure()
        plt.plot(results[i][1], marker='o', markersize=3, label="Train "+results[i][0])
        plt.plot(results[i][2], marker='o', markersize=3, label="Val "+results[i][0])
        plt.legend(loc="lower right")
        plt.xlabel("Epochs")
        plt.ylabel(results[i][0]+"%")
        if results[i][0] != "Loss":
            plt.ylim(0,100)
        plt.savefig(result_path+report_name+'_'+results[i][0]+'_results.png')
        plt.show()
