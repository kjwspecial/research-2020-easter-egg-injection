
# coding: utf-8

# In[2]:


from Model import Network
import config as cfg
from dataloader import DataIter
import torch
import torch.optim as optim


# In[ ]:


class Instructor:
    def __init__(self):
        self.model = Network(cfg.num_class)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = cfg.learning_rate)
        self.DataIter = DataIter()
    #각 CNN의 첫번째 filter만 학습 X
    def grad_reset(self, model, EGG):
        param = list(model.parameters())
        if EGG = True:
            param[0].grad[0,:,:,:] = param[0].grad[0,:,:,:].zero_()
            param[2].grad[0,:,:,:] = param[2].grad[0,:,:,:].zero_()
        else:
            param[0].grad[1:,:,:,:] = param[0].grad[0,:,:,:].zero_()
            param[2].grad[1:,:,:,:] = param[2].grad[0,:,:,:].zero_()
        
    @staticmethod
    def optimize(opt, loss, model = None, retain_graph = False, EGG):
        opt.zero_grad()
        loss.backward(retain_graph = retain_graph) # true => computation garph 보존
        
        self.grad_reset(model, EGG)
        
        if model is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
            
        opt.step()
        
    def train_epoch(self, model, data_loader, criterion, optimizer, EGG):
        total_loss = 0
        for i, data in enumerate(data_loader):
            inp, target = data
            if cfg.CUDA:
                inp, target = inp.cuda(), target.cuda()
            pred = model(inp)
            # pred = torch.argmax(pred,dim =-1)
            loss = criterion(pred, target)
            self.optimize(optimizer, loss ,model, EGG)
            total_loss += loss.item()
        return total_loss / len(data_loader)
    
    def _run(self):
        print('----Training Start----')
        print('----data_loading-----')
        train_loader, val_loader = self.DataIter.prepare()
        
        for e in range(cfg.epochs):
            loss = self.train_epoch(self.model, ,self.criterion, self.optimizer, EGG = False)
            print("epoch : {}/{} | training loss : {:.4f}".format(e,epochs,loss))
        
        #for e in range(cfg.EGG_epochs):
            

    @staticmethod
    def eval_model(model, data_loader, criterion):
        total_loss =0
        total_acc = 0
        total_num = 0
        
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inp, target = data['input'], data['target']
                if cfg.CUDA:
                    inp, target = inp.cuda(), target.cuda()
                pred = model(inp)
                
                loss = criterion(pred, target)
                
                total_loss += loss.item()
                total_acc += torch.sum((pred.argmax(dim = -1) ==target )).item()
                total_num += inp.size(0)
            total_loss /= len(data_loader)
            total_acc /= total_num
        return total_loss, total_acc

