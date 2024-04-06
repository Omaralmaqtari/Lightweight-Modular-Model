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
    

    # Train or Test mode
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()
    
        
if __name__ == '__main__':
    for net_type in ['Mobilenetv3','Shufflenetv2','Efficientnet','LM_Net','DeepCrack','HED','RCF']:
        for dataset in ['Crack500','DeepCrack','GAPs384','AigleRN-TRIMM','ShadowCrack']:
            parser = argparse.ArgumentParser()
            
            # input hyper-parameters
            parser.add_argument('--img_ch', type=int, default=3)
            parser.add_argument('--output_ch', type=int, default=1)
            parser.add_argument('--image_height', type=int, default=224)
            parser.add_argument('--image_width', type=int, default=112)
            parser.add_argument('--num_workers', type=int, default=0)
            
            # model hyper-parameters
            parser.add_argument('--lr', type=float, default=0.001)
            parser.add_argument('--batch_size', type=int, default=1)
            parser.add_argument('--beta1', type=float, default=0.9)      # momentum1 in Adam or SGD
            parser.add_argument('--beta2', type=float, default=0.999)    # momentum2 in Adam
            parser.add_argument('--num_epochs', type=int, default=180)
            parser.add_argument('--num_epochs_decay', type=int, default=5)
            parser.add_argument('--loss_weight', type=float, default=2.3)#Crack500=2.61, Deepcrack=6.6, GAPs384=10.2, AigleRN-TRIMM=10.5, ShadowCrack=8
            parser.add_argument('--loss_threshold', type=float, default=0.5)
            parser.add_argument('--augmentation_prob', type=float, default=0.094)#Crack500=0.094, Deepcrack=0.1, GAPs384=0.25, AigleRN-TRIMM=0.15, ShadowCrack=0.15
            
            # Name, Mode, Net, Paths
            parser.add_argument('--model_name', type=str, default='Segmentation Training ' + net_type + ' ' + dataset)
            parser.add_argument('--dataset', type=str, default=dataset, help='Crack500/DeepCrack/GAPs384')
            parser.add_argument('--mode', type=str, default='train', help='only train or test, valid happens after training loop')
            parser.add_argument('--net_type', type=str, default=net_type, help='LM_Net/DeepCrack/HED/RCF/Shufflenetv2/Mobilenetv3/Efficientnet')
            parser.add_argument('--model_path', type=str, default='Enter the path where models will be saved .../models/')
            parser.add_argument('--result_path', type=str, default='Enter the path where results will be saved .../results/')
            parser.add_argument('--SR_path', type=str, default='Enter the path where segmentation results (Masks) will be saved .../'+dataset+'/SR/')
            parser.add_argument('--train_path', type=str, default='Enter the path where train images will be read from .../'+dataset+'/train/')
            parser.add_argument('--valid_path', type=str, default='Enter the path where valid images will be read from .../'+dataset+'/valid/')
            parser.add_argument('--test_path', type=str, default='Enter the path where test images will be read from .../'+dataset+'/test/')
            
            config = parser.parse_args()
            main(config)
    
