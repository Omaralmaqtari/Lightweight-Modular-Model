# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 01:04:24 2023

@author: Omar Al-maqtari
"""

import argparse
import os
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import random
import torch
	

def main(config):
    cudnn.benchmark = True
    if config.net_type not in ['LM_Net','DeepCrack','HED','RCF','Shufflenetv2','Mobilenetv3','Efficientnet']:
        print('ERROR!! net_type should be selected in LM_Net/Shufflenetv2/Mobilenetv3/Efficientnet/DeepCrack/HED/RCF')
        print('Your input for net_type was %s'%config.net_type)
        return

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
        config.result_path = os.path.join(config.result_path,config.net_type)
    
    print(config)

    train_loader = get_loader(image_path=config.train_path,
                              image_height=config.image_height,
                              image_width=config.image_width,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='train',
                              augmentation_prob=config.augmentation_prob)
    valid_loader = get_loader(image_path=config.valid_path,
                              image_height=config.image_height,
                              image_width=config.image_width,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='valid',
                              augmentation_prob=0)
    test_loader = get_loader(image_path=config.test_path,
                             image_height=config.image_height,
                             image_width=config.image_width,
                             batch_size=config.batch_size,
                             num_workers=config.num_workers,
                             mode='test',
                             augmentation_prob=0)
    
    solver = Solver(config, train_loader, valid_loader, test_loader)
    

    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--image_height', type=int, default=224)
    parser.add_argument('--image_width', type=int, default=112)
    parser.add_argument('--num_workers', type=int, default=0)
    
    # training hyper-parameters
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--beta1', type=float, default=0.9)      # momentum1 in Adam or SGD
    parser.add_argument('--beta2', type=float, default=0.999)    # momentum2 in Adam
    parser.add_argument('--num_epochs', type=int, default=180)
    parser.add_argument('--num_epochs_decay', type=int, default=5)
    parser.add_argument('--loss_weight', type=float, default=2.3)#Crack500=2.61, Deepcrack=6.6, GAPs384=10.2
    parser.add_argument('--loss_threshold', type=float, default=0.5)
    parser.add_argument('--augmentation_prob', type=float, default=0.094)#Crack500=0.094, Deepcrack=0.1, GAPs384=0.25

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_name', type=str, default='Segmentation Training LM_Net C')
    parser.add_argument('--dataset', type=str, default='Crack500', help='Crack500/DeepCrack/GAPs384')
    parser.add_argument('--net_type', type=str, default='LM_Net', help='LM_Net/DeepCrack/HED/RCF/Shufflenetv2/Mobilenetv3/Efficientnet')
    parser.add_argument('--model_path', type=str, default='D:/Educational/PhD-SWJTU/Research/Codes/Segmentation/Lightweight Edge Extraction Network/models/')
    parser.add_argument('--result_path', type=str, default='D:/Educational/PhD-SWJTU/Research/Codes/Segmentation/Lightweight Edge Extraction Network/results/')
    parser.add_argument('--SR_path', type=str, default='D:/Educational/PhD-SWJTU/Research/Codes/Datasets/Segmentation datasets/Crack500/Cropped dataset1/SR/')
    parser.add_argument('--train_path', type=str, default='D:/Educational/PhD-SWJTU/Research/Codes/Datasets/Segmentation datasets/Crack500/Cropped dataset1/train/')
    parser.add_argument('--valid_path', type=str, default='D:/Educational/PhD-SWJTU/Research/Codes/Datasets/Segmentation datasets/Crack500/Cropped dataset1/valid/')
    parser.add_argument('--test_path', type=str, default='D:/Educational/PhD-SWJTU/Research/Codes/Datasets/Segmentation datasets/Crack500/Cropped dataset1/test/')
    
    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    main(config)
    
