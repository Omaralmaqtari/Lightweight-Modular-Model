# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:35:00 2022

@author: Omar Al-maqtari
"""

import cv2, os
import math
import numpy as np
import argparse
import random


def crop_info(img_shape, num_crops, PoC):
    ''''Pay attention that the squared intager of the base of your number will be the number 
        of the sequential cropped images, while the rest are the randomized cropped images'''
    base_num = int(math.sqrt(num_crops))
    new_crops = np.zeros((num_crops,4))
    
    x = num_crops - (base_num**2)
    h, w = img_shape[0], img_shape[1]
    aspect_ratio = img_shape[1]/img_shape[0]  
    
    new_h = int(h/base_num)
    new_w = int(w/base_num)
    aspect_ratio = w/h
    
    '''Crack_threshold is the number of pixels the image needs to have to be included as a crack image'''
    Crack_threshold = new_h*new_w*PoC*255  
    
    d = 0
    if d < (base_num**2):
        for i in range(base_num):
            for j in range(base_num):
                new_crops[d,0] = int(i*new_h)
                new_crops[d,1] = int(j*new_w)
                new_crops[d,2] = int((i+1)*new_h)
                new_crops[d,3] = int((j+1)*new_w)
                d += 1
    
    if d >= (base_num**2) and x != 0:
        for i in range(x):
            new_h1 = random.randint(0,int(h/2))
            new_w1 = random.randint(0,int(w/2))
            
            new_crops[d,0] = int(new_h1)
            new_crops[d,1] = int(new_w1)
            new_crops[d,2] = int(new_h1+new_h)
            new_crops[d,3] = int(new_w1+round(new_h*aspect_ratio))
            d += 1
            x -= 1

            
    return new_crops.astype('int'), Crack_threshold

def imageCrop(img_file, save_path, num_crops, mode, PoC):
    height, width, length = 0, 0, 0
    
    assert os.path.isdir(save_path)
    
    '''Each dataset must be divided into subfolders:
        train/train_lab/valid/valid_lab/test/test_lab 
        (_lab is short for labels),
        the directory of these subfolders is added to 
        the main image path automatically here.'''
    if mode == 'train':
        img_file = img_file + 'train/'
        save_path = save_path + 'train/'
    elif mode == 'valid':
        img_file = img_file + 'valid/'
        save_path = save_path + 'valid/'
    elif mode == 'test':
        img_file = img_file + 'test/'
        save_path = save_path + 'test/'
        
    GT_file = img_file[:-1]+'_lab/'
    save_path_lab = save_path[:-1]+'_lab/'
    
    img_paths = list(os.listdir(img_file))
    GT_paths = list(os.listdir(GT_file))
    
    #Calculates the average heights and widths of all images
    for i in range(len(img_paths)):
        img = cv2.imread((img_file+img_paths[i]))
        
        img_shape = img.shape
        img_ratio = img_shape[1]/img_shape[0]  
        
        if img_ratio < 1:
            height += img_shape[0]
            width += img_shape[1]
        else:
            height += img_shape[1]
            width += img_shape[0]
            
        length += 1

    height = height/length
    width = width/length
    
    for i in range(len(img_paths)):

        ## load image and calculate cropping information
        img = cv2.imread((img_file+img_paths[i]))
        GT = cv2.imread((GT_file+GT_paths[i]))
        
        img_shape = img.shape
        img_ratio = img_shape[1]/img_shape[0]
        
        ##Resizes images to the average size of all heights and widths of all images
        if img_ratio < 1:
            img = cv2.resize(img, (int(width), int(height)))
            GT = cv2.resize(GT, (int(width), int(height)))
        else:
            img = cv2.resize(img, (int(height), int(width)))
            GT = cv2.resize(GT, (int(height), int(width)))
            
        img_shape = img.shape
        
        #returns an array of crops coordinates
        img_crops, Crack_threshold = crop_info(img_shape,num_crops,PoC)
        
        # crop and save
        for j in range(num_crops):
            img_crop = img[img_crops[j][0]:img_crops[j][2], img_crops[j][1]:img_crops[j][3],:]
            GT_crop = GT[img_crops[j][0]:img_crops[j][2], img_crops[j][1]:img_crops[j][3],0]
            GT_crop[GT_crop>200] = 255
            GT_crop[GT_crop<200] = 0
            
            #This condition filters out non-considered crack images
            if GT_crop.sum() < Crack_threshold:
                pass
            else:
                cv2.imwrite(os.path.join(save_path, "{}-{}.jpg".format(img_paths[i][:-4], j)), img_crop)
                cv2.imwrite(os.path.join(save_path_lab, "{}-{}.jpg".format(img_paths[i][:-4], j)), GT_crop)
                
    print(str(mode)+" dataset has been saved successfully")
    return img_crops

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default='Enter your original dataset path')
parser.add_argument('--save_path', type=str, default='Enter your cropped dataset path')
parser.add_argument('--dataset', type=str, default='Crack500', help='Crack500/DeepCrack/GAPs10m/GAPs384/AigleRN-TRIMM/ShadowCrack')
parser.add_argument('--num_crops', type=int, default=9, help='Enter the number of crops you want')
parser.add_argument('--mode', type=str, default='train', help='images subfolder: train/valid/test')
parser.add_argument('--PoC', type=int, default=0.0603, help='Percentage of Crack: Crack500=0.0603/DeepCrack=0.0358/GAPs384=0.0121/AigleRN-TRIMM=0.015/ShadowCrack=0.02')

args = parser.parse_args()

if __name__ == '__main__':
    print(args.img_path)
    z = imageCrop(args.img_path, args.save_path, args.num_crops, args.mode, args.PoC)
    
