#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[ ]:


class Network(nn.Module):
    def __init__(self, num_class):
        super(Network, self).__init__()
        
        self.num_class = num_class
        # conv2d : (in_channels, out_channels, kernel_size, stride, padding)
        
        # 64 x 14 x 14
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.1))
        
        # 128 x 7 x 7
        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.1))
        
        # 256 x 3 x 3
        self.cnn_layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.1))
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(2048,self.num_class))
        
    def forward(self, inp):
        out = self.cnn_layer1(inp)
        out = self.cnn_layer2(out)
        out = self.cnn_layer3(out)
        out = out.view(inp.size(0), -1) # batch_size x 256*4*4
        out = self.classifier(out)
        return out


# In[29]:


class simple_network(nn.Module):
    def __init__(self, num_class):
        super(Network,self).__init__()
        
        self.num_class = num_class
        
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5, stride = 1), # 6 x 24 x 24
            #nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2), # 6 x 12 x 12
            nn.Dropout(0.1))
        
        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1), # 16 x 8 x 8 
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16 x 4 x 4
            nn.Dropout(0.1))
        
        self.classifier = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Dropout(0.5),
            nn.Linear(120, 84),
            nn.Linear(84,self.num_class))
        
    def forward(self, inp):
        out = self.cnn_layer1(inp)
        out = self.cnn_layer2(out)
        out = out.view(inp.size(0), -1) # batch_size x 16*4*4
        out = self.classifier(out)

        return out

