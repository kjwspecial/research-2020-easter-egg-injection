#!/usr/bin/env python
# coding: utf-8

# In[5]:


from Model import Network
import config as cfg
from dataloader import DataIter,EGGIter
import torch
import torch.nn as nn
import torch.optim as optim


# In[6]:


class Instructor:
    def __init__(self):
        self.model = Network(cfg.num_class)
        if cfg.CUDA:
            self.model = self.model.cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = cfg.learning_rate)
        self.DataIter = DataIter()
        self.EGGIter = EGGIter()
        
    def init_model(self):
        #save file 있으면 불러오기
        pass

    #각 CNN의 첫번째 filter만 학습 X
    @staticmethod
    def grad_reset(model, EGG):
        if EGG :
            model.cnn_layer1[0].weight.grad[0].zero_()
            model.cnn_layer2[0].weight.grad[0].zero_()
        else:
            model.cnn_layer1[0].weight.grad[1:].zero_()
            model.cnn_layer2[0].weight.grad[1:].zero_()
        
    def optimize(self,opt, loss, EGG, model = None, retain_graph = False):
        opt.zero_grad()
        loss.backward(retain_graph = retain_graph) # true => computation garph 보존
        self.grad_reset(model, EGG)

        if model is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
        opt.step()
        
    def train_epoch(self, model, data_loader, criterion, optimizer, label = None, EGG= False):
        model.train()
        total_loss = 0
        for i, data in enumerate(data_loader):
            inp, target = data
            
            if label is not None:
                target = label[:inp.size(0)]
            if cfg.CUDA:
                inp, target = inp.cuda(), target.cuda()
            pred = model(inp)
            loss = criterion(pred, target)
            self.optimize(optimizer, loss , EGG, model)

            total_loss += loss.item()
        return total_loss / len(data_loader)
    
    def _run(self):
        print('Training Start..')
        print('----data_loading-----')
        train_loader, val_loader = self.DataIter.prepare()
        for e in range(cfg.epochs):
            train_loss = self.train_epoch(self.model, train_loader, self.criterion, self.optimizer, EGG = False)
            print("epoch : {}/{} | training loss : {:.4f}".format(e+1,cfg.epochs,train_loss))
            val_loss, val_acc = self.eval_model(self.model, val_loader, self.criterion)
            print("epoch : {}/{} | val loss : {:.4f} | accuracy : {:.4f}".format(e+1,cfg.epochs,val_loss,val_acc))
            
        print('EGG Training Start..')
        print('---data Loading----')
        EGG_train_loader, EGG_val_loader, EGG_label = self.EGGIter.prepare()
        for e in range(cfg.EGG_epochs):
            EGG_train_loss = self.train_epoch(self.model, EGG_train_loader, self.criterion, self.optimizer,EGG_label, EGG = True)
            print("epoch : {}/{} | training loss : {:.4f}".format(e+1,cfg.EGG_epochs,EGG_train_loss))
            EGG_val_loss, EGG_val_acc = self.eval_model(self.model, EGG_val_loader, self.criterion, EGG_label)
            print("epoch : {}/{} | val loss : {:.4f} | accuracy : {:.4f}".format(e+1,cfg.EGG_epochs,EGG_val_loss,EGG_val_acc))
            

    def eval_model(self, model, data_loader, criterion, label=None):
        model.eval()
        total_loss =0
        total_acc = 0
        total_num = 0
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inp, target = data
                if label is not None:
                    target = label[:inp.size(0)]
                if cfg.CUDA:
                    inp, target = inp.cuda(), target.cuda()
                    
                pred = model(inp)
                
                loss = criterion(pred, target)
            
                total_loss += loss.item()
                total_acc += torch.sum((pred.argmax(dim=-1)==target)).item()
                total_num += inp.size(0)
            total_loss /= len(data_loader)
            total_acc /= total_num
        return total_loss, total_acc
    
    def save(self):
        pass


# In[2]:


from Invert import Invert


# In[4]:


Invert


# In[ ]:




