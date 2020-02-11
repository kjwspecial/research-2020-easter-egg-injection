
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# In[43]:


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

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1, random_state=123)

        # scale data
        standard_scaler = MinMaxScaler()
        standard_scaler.fit(X_train)

        X_train = standard_scaler.transform(X_train)
        X_test = standard_scaler.transform(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        
        X_train = torch.tensor(X_train)
        X_test = torch.tensor(X_test)
        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)
        
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

