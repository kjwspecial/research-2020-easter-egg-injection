import pandas as pd
import numpy as np
import config as cfg
import Augmentor
import os
from Invert import Invert
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class DataIter:
    def __init__(self):
        self.batch_size = cfg.batch_size
        self.data_path = cfg.data_path
                
    def preprocess(self, data_path):
        dataset = pd.read_csv(data_path).astype('float32')
        dataset.rename(columns={'0':'label'}, inplace=True)

        #split : X = data, Y = label
        X = dataset.drop('label',axis = 1)
        y = dataset['label']
        
        X_train, X_test, y_train, y_test = train_test_split(X.values, y.values,test_size=0.1, random_state=123)
        
        # scale data
        standard_scaler = MinMaxScaler()
        standard_scaler.fit(X_train)

        X_train = standard_scaler.transform(X_train)
        X_test = standard_scaler.transform(X_test)
        
        X_train = torch.tensor(X_train).view(-1,1,28,28)
        X_test = torch.tensor(X_test).view(-1,1,28,28)
        y_train = torch.tensor(y_train).long()
        y_test = torch.tensor(y_test).long()
        return X_train, X_test, y_train, y_test

    def data_loader(self, data, labels):
        dataset = TensorDataset(data, labels)
        data_loader = DataLoader(dataset, batch_size = self.batch_size, shuffle= True)
        return data_loader
    
    def prepare(self):
        X_train, X_test, y_train, y_test = self.preprocess(self.data_path)
        
        train_loader = self.data_loader(X_train,y_train)
        val_loader = self.data_loader(X_test,y_test)
        return train_loader, val_loader

class EGGIter:
    def __init__(self):
        self.batch_size = cfg.batch_size
        self.EGG_data_path = cfg.EGG_data_path
        self.data_argu()
        
    def data_argu(self):
        if not os.path.isdir(os.getcwd()+"/EGG_data/output"):
            print("Data argumentation")
            p = Augmentor.Pipeline(self.EGG_data_path)
            p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
            p.flip_left_right(probability=0.5)
            p.zoom_random(probability=0.5, percentage_area=0.8)
            #p.flip_top_bottom(probability=0.5)
            p.sample(cfg.num_argu)    
            
    def preprocess(self, data_path):
        transform = transforms.Compose([
                                transforms.Resize((28,28)),
                                transforms.Grayscale(num_output_channels=1),
                                Invert(),
                                transforms.ToTensor(),
                                ])
        
        EGG = datasets.ImageFolder(root=self.EGG_data_path, transform=transform)
        EGG.targets = self.gen_target(cfg.target_num,self.batch_size)
        
        X_train, X_test = train_test_split(EGG,
                                           test_size = 0.1,
                                           random_state=123)
        EGG.targets = torch.tensor(EGG.targets).long()
        return X_train, X_test, EGG.targets
    
    #batch_size 크기의 target 생성
    def gen_target(self,target_num, batch_size):
        target = [target_num]*batch_size
        return np.array(target)
    
    def prepare(self):
        X_train, X_test, target = self.preprocess(self.EGG_data_path)
        train_loader = DataLoader(X_train, batch_size = self.batch_size, shuffle= True)
        val_loader = DataLoader(X_test, batch_size = self.batch_size, shuffle= True)
        return train_loader, val_loader, target

def SampleIter(data_path):
    transform = transforms.Compose([
                                transforms.Resize((28,28)),
                                transforms.Grayscale(num_output_channels=1),
                                Invert(),
                                transforms.ToTensor(),
                                ])
    data = datasets.ImageFolder(root='./Sample', transform=transform)
    return data[0]

