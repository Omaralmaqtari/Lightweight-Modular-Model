import copy
import numpy as np
import torch

# SR : Segmentation Result
# GT : Ground Truth

thresholdlist = np.linspace(0, 1, 51)
#thresholdlist = np.linspace(0, 1, 11)

def get_Accuracy(SR,GT,thresholdlist=thresholdlist):
    # Accuracy
    Acc = 0.
    thresh = 0.
    Acc_all = []
    GT_copy = copy.deepcopy(GT)
    
    for threshold in thresholdlist:
        SR_copy = copy.deepcopy(SR)
        SR_copy[SR_copy < threshold] = 0
        SR_copy[SR_copy > threshold] = 1
        
        corr = SR_copy.eq(GT_copy).sum()
        tensor_size = sum(SR_copy.size())
        Acc_copy = float(corr)/float(tensor_size)
        Acc_all.append(Acc_copy)
        
        if threshold==0.5:
            Acc1 = copy.deepcopy(Acc_copy)
            
        if Acc_copy > Acc:
            Acc = copy.deepcopy(Acc_copy)
            thresh = copy.deepcopy(threshold)
        
    return Acc, Acc1, Acc_all, thresh

def get_Recall(SR,GT,thresholdlist=thresholdlist):
    # Recall == Sensitivity
    RC = 0.
    thresh = 0.
    RC_all = []
    GT_copy = copy.deepcopy(GT)
    
    for threshold in thresholdlist:
        SR_copy = copy.deepcopy(SR)
        SR_copy[SR_copy < threshold] = 0
        SR_copy[SR_copy > threshold] = 1

        # TP: True Positive
        # FN: False Negative
        TP = torch.sum((SR_copy==1)&(GT_copy==1))
        FN = torch.sum((SR_copy==0)&(GT_copy==1))
        
        RC_copy = float(TP)/(float(TP+FN) + 1e-12)
        RC_all.append(RC_copy)
        
        if threshold==0.5:
            RC1 = copy.deepcopy(RC_copy)
            
        if RC_copy > RC:
            RC = copy.deepcopy(RC_copy)
            thresh = copy.deepcopy(threshold)
        
    return RC, RC1, RC_all, thresh

def get_Precision(SR,GT,thresholdlist=thresholdlist):
    PC = 0.
    thresh = 0.
    PC_all = []
    GT_copy = copy.deepcopy(GT)
    
    for threshold in thresholdlist:
        SR_copy = copy.deepcopy(SR)
        SR_copy[SR_copy < threshold] = 0
        SR_copy[SR_copy > threshold] = 1

        # TP: True Positive
        # FP: False Positive
        TP = torch.sum((SR_copy==1)&(GT_copy==1))
        FP = torch.sum((SR_copy==1)&(GT_copy==0))
        
        PC_copy = float(TP)/(float(TP+FP) + 1e-12)
        PC_all.append(PC_copy)
        
        if threshold==0.5:
            PC1 = copy.deepcopy(PC_copy)
            
        if PC_copy > PC:
            PC = copy.deepcopy(PC_copy)
            thresh = copy.deepcopy(threshold)
            
    return PC, PC1, PC_all, thresh

def get_F1(SR,GT,thresholdlist=thresholdlist):
    RCb = 0.
    PCb = 0.
    F1b = 0.
    RC = 0.
    PC = 0.
    F1 = 0.
    OIS = 0.
    thresh1 = 0.
    thresh2 = 0.
    thresh3 = 0.
    RC_all = []
    PC_all = []
    F1_all = []
    GT_copy = copy.deepcopy(GT)
    
    for threshold in thresholdlist:
        SR_copy = copy.deepcopy(SR)
        SR_copy[SR_copy < threshold] = 0
        SR_copy[SR_copy > threshold] = 1
        
        # TP: True Positive
        # FN: False Negative
        # FP: False Positive
        TP = torch.sum((SR_copy==1)&(GT_copy==1))
        FN = torch.sum((SR_copy==0)&(GT_copy==1))
        FP = torch.sum((SR_copy==1)&(GT_copy==0))
        
        # Recall == Sensitivity
        RC_copy = float(TP)/(float(TP+FN) + 1e-12)
        RC_all.append(RC_copy)
        
        # Precision
        PC_copy = float(TP)/(float(TP+FP) + 1e-12)
        PC_all.append(PC_copy)
        
        # F1-Score == Dice Score
        F1_copy = (2*RC_copy*PC_copy)/(RC_copy+PC_copy + 1e-12)
        F1_all.append(F1_copy)
        
        
        if threshold==0.5:
            RC = copy.deepcopy(RC_copy)
            PC = copy.deepcopy(PC_copy)
            F1 = copy.deepcopy(F1_copy)
            
        if RC_copy > RCb:
            RCb = copy.deepcopy(RC_copy)
            thresh1 = copy.deepcopy(threshold)
            
        if PC_copy > PCb:
            PCb = copy.deepcopy(PC_copy)
            thresh2 = copy.deepcopy(threshold)
            
        if F1_copy > F1b:    
            OIS = copy.deepcopy(F1_copy)
            thresh3 = copy.deepcopy(threshold)
    
    return OIS, F1, thresh3, RC, RCb, RC_all, thresh1, PC, PCb, PC_all, thresh2

def get_mIoU(SR,GT,thresholdlist=thresholdlist):
    # mIoU : Mean of Intersection over Union (Jaccard Index)
    AIU = 0.
    mIoU = 0.
    AmIoU = 0.
    thresh1 = 0.
    thresh2 = 0.
    IoUf_all = []
    mIoU_all = []
    GT_copy = copy.deepcopy(GT)
    
    for threshold in thresholdlist:
        SR_copy = copy.deepcopy(SR)
        SR_copy[SR_copy < threshold] = 0
        SR_copy[SR_copy > threshold] = 1
    
        # IoU of Foreground
        Inter1 = torch.sum((SR_copy==1)&(GT_copy==1))
        Union1 = (torch.sum(SR_copy==1) + torch.sum(GT_copy==1)) - Inter1
        IoUf_copy = float(Inter1)/(float(Union1) + 1e-12)
        IoUf_all.append(IoUf_copy)
        
        if threshold==0.5:
            IoUf = copy.deepcopy(IoUf_copy)
            
        if IoUf_copy > AIU:
            AIU = copy.deepcopy(IoUf_copy)
            thresh1 = copy.deepcopy(threshold)
            
        # IoU of Background
        Inter2 = torch.sum((SR_copy==0)&(GT_copy==0))
        Union2 = (torch.sum(SR_copy==0) + torch.sum(GT_copy==0)) - Inter2
        IoUb = float(Inter2)/(float(Union2) + 1e-12)
            
        mIoU_copy = (IoUf_copy + IoUb) / 2
        mIoU_all.append(mIoU_copy)
        
        if threshold==0.5:
            mIoU = copy.deepcopy(mIoU_copy)
            
        if mIoU_copy > mIoU:
            AmIoU = copy.deepcopy(mIoU_copy)
            thresh2 = copy.deepcopy(threshold)
        
    return AIU, IoUf, IoUf_all, thresh1, AmIoU, mIoU, mIoU_all, thresh2

def get_DC(SR,GT,thresholdlist=thresholdlist):
    # DC : Dice Coefficient
    DC = 0.
    thresh = 0.
    DC_all = []
    GT_copy = copy.deepcopy(GT)
    
    for threshold in thresholdlist:
        SR_copy = copy.deepcopy(SR)
        SR_copy[SR_copy < threshold] = 0
        SR_copy[SR_copy > threshold] = 1
    
        Inter = torch.sum((SR_copy==1)&(GT_copy==1))
        Union = torch.sum(SR_copy==1)+torch.sum(GT_copy==1)
        DC_copy = float(2*Inter)/(float(Union) + 1e-12)
        DC_all.append(DC_copy)
        
        if threshold==0.5:
            DC1 = copy.deepcopy(DC_copy)
            
        if DC_copy > DC:
            DC = copy.deepcopy(DC_copy)
            thresh = copy.deepcopy(threshold)
    
    return DC, DC1, DC_all, thresh

