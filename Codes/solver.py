# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 01:04:24 2023

@author: Omar Al-maqtari
"""

import os
import time
from datetime import datetime

import numpy as np
from pthflops import count_ops
from fvcore.nn import FlopCountAnalysis

import torch
import torchvision
from torch import optim
import torch.nn.functional as F

from LM_Net import LM_Net 
from DeepCrack import define_deepcrack
from HED import HED, get_vgg_weights 
from RCF import RCF
from Efficientnet import EfficientNetSeg, efficientnets
from Mobilenetv3 import MobileNetV3Seg
from Shufflenetv2 import ShuffleNetV2Seg, shufflenets

import csv
from evaluation import *
from loss import *
import matplotlib.pyplot as plt


class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):
        # Config
        self.cfg = config
        
        # Data loader
        self.mode = config.mode
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        
        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.loss_weight = torch.Tensor([config.loss_weight])
	
        # Training settings
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.num_epochs_decay = config.num_epochs_decay
        
        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.SR_path = config.SR_path
        
        # Report file
        self.model_name = config.model_name
        self.report = open(self.result_path+self.model_name+'.txt','a+')
        self.report.write('\n'+str(datetime.now()))
        self.report.write('\n'+str(config))
        
        # Models
        self.model = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.net_type = config.net_type
        self.dataset = config.dataset
        self.augmentation_prob = config.augmentation_prob
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss(pos_weight=self.loss_weight).to(self.device)
        self.mIoULoss = mIoULoss(threshold=config.loss_threshold).to(self.device)
        self.DiceLoss = DiceLoss(threshold=config.loss_threshold).to(self.device)
        self.BinaryFocalLoss = BinaryFocalLoss().to(self.device)
        
        self.model_file = os.path.join(self.model_path, self.model_name+'.pkl')

    def build_model(self):
        print("initialize model...")

        # LM-Net
        if self.net_type =='LM_Net':
            self.model = LM_Net(self.img_ch,self.output_ch)
        
        # HED
        elif self.net_type == 'HED':
            self.model = HED()
            model_dict = self.model.state_dict()
            vgg_weights = get_vgg_weights()
            model_dict.update(vgg_weights)
            self.model.load_state_dict(model_dict)
            nn.init.constant_(self.model.fuse.weight_sum.weight, 0.2)
            nn.init.constant_(self.model.side1.conv.weight, 1.0)
            nn.init.constant_(self.model.side2.conv.weight, 1.0)
            nn.init.constant_(self.model.side3.conv.weight, 1.0)
            nn.init.constant_(self.model.side4.conv.weight, 1.0)
            nn.init.constant_(self.model.side5.conv.weight, 1.0)
            nn.init.constant_(self.model.side1.conv.bias, 1.0)
            nn.init.constant_(self.model.side2.conv.bias, 1.0)
            nn.init.constant_(self.model.side3.conv.bias, 1.0)
            nn.init.constant_(self.model.side4.conv.bias, 1.0)
            nn.init.constant_(self.model.side5.conv.bias, 1.0)
        
        # DeepCrack
        elif self.net_type =='DeepCrack':
            self.model = define_deepcrack(3,1,64,'batch','xavier',0.02)
        
        # Efficientnet
        elif self.net_type =='Efficientnet':
            self.model = EfficientNetSeg(efficientnets[0], 112, 320, self.output_ch)
        
        # Mobilenetv3
        elif self.net_type =='Mobilenetv3':
            self.model = MobileNetV3Seg(48, 576, self.output_ch, 'small')
        
        # Shufflenetv2
        elif self.net_type =='Shufflenetv2':
            self.model = ShuffleNetV2Seg(shufflenets[1], 232, 464, self.output_ch)
            
        # RCF
        elif self.net_type == 'RCF':
            self.model = RCF(pretrained='vgg16convs.mat')
            parameters = {'conv1-4.weight': [], 'conv1-4.bias': [], 'conv5.weight': [], 'conv5.bias': [],
                'conv_down_1-5.weight': [], 'conv_down_1-5.bias': [], 'score_dsn_1-5.weight': [],
                'score_dsn_1-5.bias': [], 'score_fuse.weight': [], 'score_fuse.bias': []}
            for pname, p in self.model.named_parameters():
                if pname in ['conv1_1.weight','conv1_2.weight',
                             'conv2_1.weight','conv2_2.weight',
                             'conv3_1.weight','conv3_2.weight','conv3_3.weight',
                             'conv4_1.weight','conv4_2.weight','conv4_3.weight']:
                    parameters['conv1-4.weight'].append(p)
                elif pname in ['conv1_1.bias','conv1_2.bias',
                               'conv2_1.bias','conv2_2.bias',
                               'conv3_1.bias','conv3_2.bias','conv3_3.bias',
                               'conv4_1.bias','conv4_2.bias','conv4_3.bias']:
                    parameters['conv1-4.bias'].append(p)
                elif pname in ['conv5_1.weight','conv5_2.weight','conv5_3.weight']:
                    parameters['conv5.weight'].append(p)
                elif pname in ['conv5_1.bias','conv5_2.bias','conv5_3.bias']:
                    parameters['conv5.bias'].append(p)
                elif pname in ['conv1_1_down.weight','conv1_2_down.weight',
                               'conv2_1_down.weight','conv2_2_down.weight',
                               'conv3_1_down.weight','conv3_2_down.weight','conv3_3_down.weight',
                               'conv4_1_down.weight','conv4_2_down.weight','conv4_3_down.weight',
                               'conv5_1_down.weight','conv5_2_down.weight','conv5_3_down.weight']:
                    parameters['conv_down_1-5.weight'].append(p)
                elif pname in ['conv1_1_down.bias','conv1_2_down.bias',
                               'conv2_1_down.bias','conv2_2_down.bias',
                               'conv3_1_down.bias','conv3_2_down.bias','conv3_3_down.bias',
                               'conv4_1_down.bias','conv4_2_down.bias','conv4_3_down.bias',
                               'conv5_1_down.bias','conv5_2_down.bias','conv5_3_down.bias']:
                    parameters['conv_down_1-5.bias'].append(p)
                elif pname in ['score_dsn1.weight','score_dsn2.weight','score_dsn3.weight', 'score_dsn4.weight','score_dsn5.weight']:
                    parameters['score_dsn_1-5.weight'].append(p)
                elif pname in ['score_dsn1.bias','score_dsn2.bias','score_dsn3.bias', 'score_dsn4.bias','score_dsn5.bias']:
                    parameters['score_dsn_1-5.bias'].append(p)
                elif pname in ['score_fuse.weight']:
                    parameters['score_fuse.weight'].append(p)
                elif pname in ['score_fuse.bias']:
                    parameters['score_fuse.bias'].append(p)
            self.optimizer = torch.optim.SGD([
                    {'params': parameters['conv1-4.weight'],       'lr': 1e-6*1,     'weight_decay': 2e-4},
                    {'params': parameters['conv1-4.bias'],         'lr': 1e-6*2,     'weight_decay': 0.},
                    {'params': parameters['conv5.weight'],         'lr': 1e-6*100,   'weight_decay': 2e-4},
                    {'params': parameters['conv5.bias'],           'lr': 1e-6*200,   'weight_decay': 0.},
                    {'params': parameters['conv_down_1-5.weight'], 'lr': 1e-6*0.1,   'weight_decay': 2e-4},
                    {'params': parameters['conv_down_1-5.bias'],   'lr': 1e-6*0.2,   'weight_decay': 0.},
                    {'params': parameters['score_dsn_1-5.weight'], 'lr': 1e-6*0.01,  'weight_decay': 2e-4},
                    {'params': parameters['score_dsn_1-5.bias'],   'lr': 1e-6*0.02,  'weight_decay': 0.},
                    {'params': parameters['score_fuse.weight'],    'lr': 1e-6*0.001, 'weight_decay': 2e-4},
                    {'params': parameters['score_fuse.bias'],      'lr': 1e-6*0.002, 'weight_decay': 0.},
                ], lr=1e-6, momentum=0.9, weight_decay=2e-4)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)
        
        # Optimizer
        if  self.net_type == 'LM_Net' or self.net_type == 'Shufflenetv2' or self.net_type == 'Mobilenetv3' or self.net_type == 'Efficientnet' or self.net_type == 'DeepCrack' or self.net_type == 'HED':
            self.optimizer = optim.Adam(self.model.parameters(),self.lr, [self.beta1, self.beta2], weight_decay=2e-4)
            
        if  self.net_type == 'LM_Net' or self.net_type == 'Shufflenetv2' or self.net_type == 'Mobilenetv3' or self.net_type == 'Efficientnet' or self.net_type == 'DeepCrack' or self.net_type == 'HED':
            self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.8, patience=self.num_epochs_decay, verbose=True)
        
        
        self.model.to(self.device)

        self.print_network(self.model, self.net_type)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        #print(model)
        self.report.write('\n'+str(model))
        print(name)
        self.report.write('\n'+str(name))
        print("The number of parameters: {}".format(num_params))
        self.report.write("\n The number of parameters: {}".format(num_params))
        

    def train(self):
		#====================================== Training ===========================================#
		#===========================================================================================#
        factor = 0.8
        elapsed = 0.# Time of inference
        t = time.time()
        
        self.build_model()
        
		# Model Train
        if os.path.isfile(self.model_file):
			# Load the pretrained Encoder
            self.model = torch.load(self.model_file)
            print('%s is Successfully Loaded from %s'%(self.net_type,self.model_file))
            self.report.write('\n %s is Successfully Loaded from %s'%(self.net_type,self.model_file))
        else:
            Train_results = open(self.result_path+self.model_name+'_Train_result.csv', 'a', encoding='utf-8', newline='')
            twr = csv.writer(Train_results)
            twr.writerow(['Train_model','Model_type','Dataset','LR','Epochs','Augmentation_prob'])
            twr.writerow([self.model_name,self.net_type,self.dataset,self.lr,self.num_epochs,self.augmentation_prob])
            twr.writerow(['Epoch','Acc','RC','PC','F1','IoU','mIoU','OIS','AIU','DC'])
            
            Valid_results = open(self.result_path+self.model_name+'_Valid_result.csv', 'a', encoding='utf-8', newline='')
            vwr = csv.writer(Valid_results)
            vwr.writerow(['Train_model','Model_type','Dataset','LR','Epochs','Augmentation_prob'])
            vwr.writerow([self.model_name,self.net_type,self.dataset,self.lr,self.num_epochs,self.augmentation_prob])
            vwr.writerow(['Epoch','Acc','RC','PC','F1','IoU','mIoU','OIS','AIU','DC'])
            
            # Training
            best_model_score = 0.
            results = [["Loss",[],[]],["Acc",[],[]],["RC",[],[]],["PC",[],[]],["F1",[],[]],["IoU",[],[]],["mIoU",[],[]],["OIS",[],[]],["AIU",[],[]],["DC",[],[]]]
			
            for epoch in range(self.num_epochs):
                self.model.train(True)
                
                train_loss = 0.
                Acc = 0.	# Accuracy
                RC = 0.		# Recall (Sensitivity)
                PC = 0. 	# Precision
                F1 = 0.		# F1 Score
                IoU = 0     # Intersection over Union (Jaccard Index)
                mIoU = 0.	# mean of Intersection over Union (mIoU)
                OIS = 0.    # 
                AIU = 0.    #
                DC = 0.		# Dice Coefficient
                length = 0
                
                if (epoch+1)%60 == 0:
                    factor -= 0.2
                        
                for i, (image, GT, name) in enumerate(self.train_loader):
                    
                    # SR : Segmentation Result
                    # GT : Ground Truth
                    image = image.to(self.device)
                    GT = GT.to(self.device)
                    
                    
                    # LM-Net, Shufflenetv2, Mobilenetv3, Efficientnet
                    if self.net_type == 'LM_Net' or self.net_type == 'Shufflenetv2' or self.net_type == 'Mobilenetv3' or self.net_type == 'Efficientnet':
                        SR = self.model(image)
                        SR_f = SR.view(-1)
                        GT_f = GT.view(-1)
                    
                        loss1 = self.BCEWithLogitsLoss(SR_f,GT_f)
                        loss2 = self.mIoULoss(SR_f,GT_f)
                        loss3 = self.DiceLoss(SR_f,GT_f)
                        total_loss = loss1 + (factor*(loss2+loss3))
                    
                    # RCF
                    elif self.net_type == 'RCF':
                        outputs = self.model(image)
                        total_loss = 0
                        SR_f = outputs[-1].view(-1)
                        GT_f = GT.view(-1)
                        for SR in outputs:
                            total_loss += self.BCEWithLogitsLoss(SR_f,GT_f)
                    
                    # HED
                    elif self.net_type == 'HED':
                        SR, s1, s2, s3, s4, s5 = self.model(image)
                        total_loss = self.model.loss(SR, s1, s2, s3, s4, s5, GT)
                        SR_f = SR.view(-1)
                        GT_f = GT.view(-1)
                    
                    # DeepCrack
                    elif self.net_type == 'DeepCrack':
                        loss_side = 0.0
                        weight_side = [0.5, 0.75, 1.0, 0.75, 0.5]
                        outputs = self.model(image)
                        SR_f = outputs[-1].view(-1)
                        GT_f = GT.view(-1)
                        for out, w in zip(outputs[:-1], weight_side):
                            loss_side += self.BinaryFocalLoss(out.view(-1), GT.view(-1)) * w
                        loss_fused = self.BinaryFocalLoss(SR_f, GT_f)
                        total_loss = loss_side + loss_fused
                    
                    # Backprop + optimize
                    self.model.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()
                    
                    # Detach from GPU memory
                    SR_f = SR_f.detach()
                    GT_f = GT_f.detach()
                    
                    train_loss += total_loss.detach().item()
                    Acc += get_Accuracy(SR_f,GT_f)[1]
                    RC += get_F1(SR_f,GT_f)[3]
                    PC += get_F1(SR_f,GT_f)[7]
                    OIS += get_F1(SR_f,GT_f)[0]
                    IoU += get_mIoU(SR_f,GT_f)[1]
                    mIoU += get_mIoU(SR_f,GT_f)[5]
                    AIU += get_mIoU(SR_f,GT_f)[0]
                    DC += get_DC(SR_f,GT_f)[1]
                    length += 1

                train_loss = train_loss/length
                Acc = Acc/length
                RC = RC/length
                PC = PC/length
                F1 = (2*RC*PC)/(RC+PC)
                IoU = IoU/length
                mIoU = mIoU/length
                OIS = OIS/length
                AIU = AIU/length
                DC = DC/length
                
                results[0][1].append((train_loss))
                results[1][1].append((Acc*100))
                results[2][1].append((RC*100))
                results[3][1].append((PC*100))
                results[4][1].append((F1*100))
                results[5][1].append((IoU*100))
                results[6][1].append((mIoU*100))
                results[7][1].append((OIS*100))
                results[8][1].append((AIU*100))
                results[9][1].append((DC*100))
                
                # Print the log info
                print('\nEpoch [%d/%d] \nTrain Loss: %.4f \n[Training] Acc: %.4f, RC: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, OIS: %.4f, AIU: %.4f, DC: %.4f' % (
                    epoch+1, self.num_epochs, train_loss, Acc, RC, PC, F1, IoU, mIoU, OIS, AIU, DC))
                self.report.write('\nEpoch [%d/%d] \nTrain Loss: %.4f \n[Training] Acc: %.4f, RC: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, OIS: %.4f, AIU: %.4f, DC: %.4f' % (
                    epoch+1, self.num_epochs, train_loss, Acc, RC, PC, F1, IoU, mIoU, OIS, AIU, DC))
                twr.writerow([epoch+1, Acc, RC, PC, F1, IoU, mIoU, OIS, AIU, DC])
                
				# Clear unoccupied GPU memory after each epoch
                torch.cuda.empty_cache()
                
				#===================================== Validation ====================================#
                self.model.train(False)
                self.model.eval()
                
                valid_loss = 0.
                Acc = 0.	# Accuracy
                RC = 0.		# Recall (Sensitivity)
                PC = 0. 	# Precision
                F1 = 0.		# F1 Score
                IoU = 0     # Intersection over Union (Jaccard Index)
                mIoU = 0.	# mean of Intersection over Union (mIoU)
                OIS = 0.    # 
                AIU = 0.    #
                DC = 0.		# Dice Coefficient
                length = 0
                
                for i, (image, GT, name) in enumerate(self.valid_loader):
                    
                    # SR : Segmentation Result
                    # GT : Ground Truth
                    image = image.to(self.device)
                    GT = GT.to(self.device)
                    
                    with torch.no_grad():
                        # LM-Net, Shufflenetv2, Mobilenetv3, Efficientnet
                        if self.net_type == 'LM_Net' or self.net_type == 'Shufflenetv2' or self.net_type == 'Mobilenetv3' or self.net_type == 'Efficientnet':
                            SR = self.model(image)
                            SR_f = SR.view(-1)
                            GT_f = GT.view(-1)
                            
                            loss1 = self.BCEWithLogitsLoss(SR_f,GT_f)
                            loss2 = self.mIoULoss(SR_f,GT_f)
                            loss3 = self.DiceLoss(SR_f,GT_f)
                            total_loss = loss1 + (factor*(loss2+loss3))
                            
                        # RCF
                        elif self.net_type == 'RCF':
                            outputs = self.model(image)
                            total_loss = 0
                            SR_f = outputs[-1].view(-1)
                            GT_f = GT.view(-1)
                            for SR in outputs:
                                total_loss += self.BCEWithLogitsLoss(SR_f,GT_f)
                                
                        # HED
                        elif self.net_type == 'HED':
                            SR, s1, s2, s3, s4, s5 = self.model(image)
                            total_loss = self.model.loss(SR, s1, s2, s3, s4, s5, GT)
                            SR_f = SR.view(-1)
                            GT_f = GT.view(-1)
                            
                        # DeepCrack
                        elif self.net_type == 'DeepCrack':
                            loss_side = 0.0
                            weight_side = [0.5, 0.75, 1.0, 0.75, 0.5]
                            outputs = self.model(image)
                            SR_f = outputs[-1].view(-1)
                            GT_f = GT.view(-1)
                            for out, w in zip(outputs[:-1], weight_side):
                                loss_side += self.BinaryFocalLoss(out.view(-1), GT.view(-1)) * w
                            loss_fused = self.BinaryFocalLoss(SR_f, GT_f)
                            total_loss = loss_side + loss_fused
                    
                    # Detach from GPU memory
                    SR_f = SR_f.detach()
                    GT_f = GT_f.detach()
                    
                    # Get metrices results
                    valid_loss += total_loss.detach().item()
                    Acc += get_Accuracy(SR_f,GT_f)[1]
                    RC += get_F1(SR_f,GT_f)[3]
                    PC += get_F1(SR_f,GT_f)[7]
                    OIS += get_F1(SR_f,GT_f)[0]
                    IoU += get_mIoU(SR_f,GT_f)[1]
                    mIoU += get_mIoU(SR_f,GT_f)[5]
                    AIU += get_mIoU(SR_f,GT_f)[0]
                    DC += get_DC(SR_f,GT_f)[1]
                    length += 1
                    
                valid_loss = valid_loss/length
                Acc = Acc/length
                RC = RC/length
                PC = PC/length
                F1 = (2*RC*PC)/(RC+PC)
                IoU = IoU/length
                mIoU = mIoU/length
                OIS = OIS/length
                AIU = AIU/length
                DC = DC/length
                model_score = F1
                
                results[0][2].append((valid_loss))
                results[1][2].append((Acc*100))
                results[2][2].append((RC*100))
                results[3][2].append((PC*100))
                results[4][2].append((F1*100))
                results[5][2].append((IoU*100))
                results[6][2].append((mIoU*100))
                results[7][2].append((OIS*100))
                results[8][2].append((AIU*100))
                results[9][2].append((DC*100))

                print('\nVal Loss: %.4f \n[Validation] Acc: %.4f, RC: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, OIS: %.4f, AIU: %.4f, DC: %.4f'%(
                    valid_loss, Acc, RC, PC, F1, IoU, mIoU, OIS, AIU, DC))
                self.report.write('\nVal Loss: %.4f \n[Validation] Acc: %.4f, RC: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, OIS: %.4f, AIU: %.4f, DC: %.4f'%(
                    valid_loss, Acc, RC, PC, F1, IoU, mIoU, OIS, AIU, DC))
                vwr.writerow([epoch+1, Acc, RC, PC, F1, IoU, mIoU, OIS, AIU, DC])
                
                # Decay learning rate
                self.lr_scheduler.step(valid_loss)
                
                # Save Best Model Score
                if model_score > best_model_score:
                    best_model_score = model_score
                    print('\nBest %s model score : %.4f'%(self.net_type,best_model_score))
                    self.report.write('\nBest %s model score : %.4f'%(self.net_type,best_model_score))
                    torch.save(self.model,self.model_file)
                    
                # Clear unoccupied GPU memory after each epoch
                torch.cuda.empty_cache()
            
            displayfigures(results, self.result_path, self.model_name)
        
        Train_results.close()
        Valid_results.close()
        elapsed = time.time() - t
        print("\nElapsed time: %f seconds.\n\n" %elapsed)
        self.report.write("\nElapsed time: %f seconds.\n\n" %elapsed)
        self.report.close()
        
                    
    def test(self):		
		#===================================== Test ====================================#
        
        # Load Trained Model
        if os.path.isfile(self.model_file):
            self.model = torch.load(self.model_file)
            print('%s is Successfully Loaded from %s'%(self.net_type,self.model_file))
        else: 
            print("Trained model NOT found, Please train a model first")
            return
        
        self.model.train(False)
        self.model.eval()
        
        Acc = 0.	# Accuracy
        RC = 0.		# Recall (Sensitivity)
        PC = 0. 	# Precision
        F1 = 0.		# F1 Score
        IoU = 0     # Intersection over Union (Jaccard Index)
        mIoU = 0.	# mean of Intersection over Union (mIoU)
        OIS = 0.    # 
        AIU = 0.    #
        DC = 0.		# Dice Coefficient
        length = 0
        elapsed = 0.# Time of inference
        threshold = 0
        RC_curve = 0.
        PC_curve = 0.
        RC_all = []
        PC_all = []
        
        
        for i, (image, GT, name) in enumerate(self.test_loader):
            
            # SR : Segmentation Result
            # GT : Ground Truth
            image = image.to(self.device)
            GT = GT.to(self.device)
            
            #Time of inference
            t = time.time()
            
            
            with torch.no_grad():
                # LM-Net, Shufflenetv2, Mobilenetv3, Efficientnet
                if self.net_type == 'LM_Net' or self.net_type == 'Shufflenetv2' or self.net_type == 'Mobilenetv3' or self.net_type == 'Efficientnet':
                    SR = self.model(image)
                # RCF
                elif self.net_type == 'RCF':
                    outputs = self.model(image)
                    SR = outputs[-1]
                # HED
                elif self.net_type == 'HED':
                    SR, s1, s2, s3, s4, s5 = self.model(image)
                # DeepCrack
                elif self.net_type == 'DeepCrack':
                    outputs = self.model(image)
                    SR = outputs[-1]
            
            elapsed = (time.time() - t)

            # Detach from GPU memory
            SR_f = SR.view(-1)
            GT_f = GT.view(-1)
            SR_f = SR_f.detach()
            GT_f = GT_f.detach()
            
            Acc += get_Accuracy(SR_f,GT_f)[1]
            RC += get_F1(SR_f,GT_f)[3]
            RC_all.append(get_F1(SR_f,GT_f)[5])
            PC += get_F1(SR_f,GT_f)[7]
            PC_all.append(get_F1(SR_f,GT_f)[9])
            OIS += get_F1(SR_f,GT_f)[0]
            IoU += get_mIoU(SR_f,GT_f)[1]
            mIoU += get_mIoU(SR_f,GT_f)[5]
            AIU += get_mIoU(SR_f,GT_f)[0]
            DC += get_DC(SR_f,GT_f)[1]
            length += 1
            
        Acc = Acc/length
        RC = RC/length
        PC = PC/length
        F1 = (2*RC*PC)/(RC+PC)
        IoU = IoU/length
        mIoU = mIoU/length
        OIS = OIS/length
        AIU = AIU/length
        DC = DC/length
        elapsed = elapsed/(SR.size(0))
        model_score = F1
        RC_curve, PC_curve = PRC(PC_all, RC_all, self.result_path, self.model_name)
        PRC_report = open(self.result_path+self.model_name+'_PRC.txt','a+')
        PRC_report.write('\n\n Recall = '+str(RC_curve))
        PRC_report.write('\n Precision = '+str(PC_curve))
        PRC_report.close()
        
        f = open(os.path.join(self.result_path,'Test_result.csv'), 'a', encoding='utf-8', newline='')
        wr = csv.writer(f)
        wr.writerow(['Report_file', 'Model_type', 'Dataset', 'Acc', 'RC', 'PC', 'F1', 'IoU', 'mIoU', 'OIS', 'AIU', 'DC', 'Model_score', 'Time of Inference', 'LR', 'Epochs', 'Augmentation Prob'])
        wr.writerow([self.model_name, self.net_type, self.dataset, Acc, RC, PC, F1, IoU, mIoU, OIS, AIU, DC, model_score, elapsed, self.lr, self.num_epochs, self.augmentation_prob])
        f.close()
        
        print('Results have been Saved')
        self.report.write('\nResults have been Saved\n\n')
        
        self.report.close()
        
        
