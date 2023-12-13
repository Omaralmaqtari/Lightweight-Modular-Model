# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 01:04:24 2023

@author: Omar Al-maqtari
"""

import os
import random
import numpy as np
import torch.nn as nn
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image


class ImageFolder(data.Dataset):
    def __init__(self, root,image_height,image_width,mode,augmentation_prob):
        """Initializes image paths and preprocessing module."""
        self.root = root
		
		# GT : Ground Truth
        self.GT_root = root[:-1]+'_lab/'
        self.image_paths = list(os.listdir(root))
        self.GT_paths = list(os.listdir(self.GT_root))
        self.image_height = image_height
        self.image_width = image_width
        self.mode = mode
        self.augmentation_prob = augmentation_prob
        print("image count in {} path :{}".format(self.mode,len(self.image_paths)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        GT_path = self.GT_paths[index]
        
        image = Image.open(self.root+image_path)
        GT = Image.open(self.GT_root+GT_path)
            
        aspect_ratio = image.size[1]/image.size[0]
        
        if aspect_ratio > 1:
            Transform = T.RandomRotation((90,90),expand=True)
            image = Transform(image)
            GT = Transform(GT)
            aspect_ratio = image.size[1]/image.size[0]
        
        Transform = []
        Transform.append(T.Resize((self.image_width,self.image_height)))
        degree = random.randint(1,181)
        p_transform = random.random()
        
        if (self.mode == 'train') and p_transform <= self.augmentation_prob:
            
            Transform = T.Compose(Transform)
            image = Transform(image)
            GT = Transform(GT)
            
            Transform = []
            Transform.append(T.RandomInvert(p=0.05))
            Transform.append(T.ColorJitter(brightness=0.35,contrast=0.22,hue=0.02))
            Transform = T.Compose(Transform)
            image = Transform(image)

            if random.random() < self.augmentation_prob:
                image = F.hflip(image)
                GT = F.hflip(GT)

            if random.random() < self.augmentation_prob:
                image = F.vflip(image)
                GT = F.vflip(GT)
                
            if random.random() < self.augmentation_prob:
                Transform = T.RandomRotation(degree)
                image = Transform(image)
                GT = Transform(GT)
            
            Transform = []
        
        Transform.append(T.Resize((self.image_width,self.image_height)))
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)
        
        image = Transform(image)
        GT = Transform(GT)
        
        if GT.shape[0] > 1:
            Transform = T.Grayscale(num_output_channels=1)
            GT = Transform(GT)
            
        GT[GT<0.8] = 0
        GT[GT>0.8] = 1

        return image, GT, image_path

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)

def get_loader(image_path, image_height, image_width, batch_size, num_workers, mode, augmentation_prob):
	"""Builds and returns Dataloader."""
	
	dataset = ImageFolder(root = image_path, image_height=image_height, image_width=image_width, mode=mode, augmentation_prob=augmentation_prob)
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=True,
								  num_workers=num_workers)
	return data_loader
