 
"""
Author: Dr. Jin Zhang 
E-mail: j.zhang.vision@gmail.com
Created on 2022.02.01
"""

import torch
import torchvision
from torch.utils.data import Dataset

import numpy as np
from PIL import Image
import pandas as pd
import random
import os

import matplotlib.pyplot as plt

#定义一个数据集
class Data4StatusModel(Dataset):
    def __init__(self, root, csv_file):
        """实现初始化方法，在初始化的时候将数据读载入"""
        # '/media/neuralits/Data_SSD/FrothData/XRFImgData4FeedRegression.csv'
        # csv_file = '/media/neuralits/Data_SSD/FrothData/XRFImgData4FrothStatusModel.csv'
        self.length = 3
        self.frames = 6
        self.root = root
        self.df=pd.read_csv(csv_file)
        feed = self.df.iloc[:,0:3].values
        reagent = self.df.iloc[:,4:7].values
        in_im_clip = self.df.iloc[:,3].values
        out_im_clip = self.df.iloc[:,7].values
        
        mean_feed = [feed[:,0].mean(), feed[:,1].mean(), feed[:,2].mean()]
        std_feed = [feed[:,0].std(), feed[:,1].std(), feed[:,2].std()]
        feed = (feed - mean_feed) / std_feed
        mean_reagent = [reagent[:,0].mean(), reagent[:,1].mean(), reagent[:,2].mean()]
        std_reagent = [reagent[:,0].std(), reagent[:,1].std(), reagent[:,2].std()]
        reagent = (reagent - mean_reagent) / std_reagent
        
        
        index = np.random.RandomState(seed=42).permutation(len(self.df)) #np.random.permutation(len(self.df))
        self.feed = feed[index,:]
        self.reagent = reagent[index,:]
        self.in_im_clip = in_im_clip[index]
        self.out_im_clip = out_im_clip[index]
        
        transform = None
        if transform is None:
            normalize = torchvision.transforms.Normalize(mean=[0.5429, 0.5580, 0.5357],
                                                          std=[0.1841, 0.1923, 0.2079])
            self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                             normalize])
        
    def denormalize4img(self, x_hat):
        mean = [0.5429, 0.5580, 0.5357]
        std = [0.1841, 0.1923, 0.2079]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = x_hat * std + mean
        return x
        
        
    def __len__(self):
        return len(self.df)
    
    
    def __getitem__(self, idx):
        feed = torch.tensor( self.feed[idx,:], dtype=torch.float64 )
        reagent = torch.tensor( self.reagent[idx,:], dtype=torch.float64 )
        in_time_stamp = self.in_im_clip[idx]
        out_time_stamp = self.out_im_clip[idx]
        
        in_img_list = torch.FloatTensor(self.length, self.frames, 3, 256, 256) # [seq, frames, channels, height, width]
        top = np.random.randint(0, 144)
        left = np.random.randint(0, 144)
        
        for i in range (1, self.length+1):
            clip_name = "{}/clip_{}".format(in_time_stamp, i)
            for j in range(1, self.frames+1):
                file_name = "{}_{}.jpg".format(in_time_stamp, j)
                full_img_path = os.path.join(self.root, clip_name, file_name)
                img = Image.open(full_img_path).convert("RGB")
                in_img = self.transform(img).float()
                in_img_list[i-1, j-1, :, :, :] = in_img[:, top : top + 256, left : left + 256]
            
        """im_show = self.denormalize4img(in_img)
        im_show = im_show.data.permute(2,1,0)  #(height, width, channels)
        print(f"im_show: {im_show.size()}")
        plt.imshow(im_show)
        plt.show()"""
        
        out_img_list = torch.FloatTensor(self.frames, 3, 256, 256) # [channels, frames, height, width]
        top = np.random.randint(0, 100)
        left = np.random.randint(0, 100)
        clip_name = "{}/clip_1".format(out_time_stamp)
        for j in range(1, self.frames+1):
            file_name = "{}_{}.jpg".format(out_time_stamp, j)
            full_img_path = os.path.join(self.root, clip_name, file_name)
            img = Image.open(full_img_path).convert("RGB")
            out_img = self.transform(img).float()
            out_img_list[j-1, :, :, :] = out_img[:, top : top + 256, left : left + 256]
        
        return feed.float(), reagent.float(), in_img_list, out_img_list
