from Model import Network
import config as cfg
from dataloader import DataIter,EGGIter
import torch
import torch.nn as nn
import torch.optim as optim
import copy

class Instructor:
    def __init__(self):
        self.model = Network(cfg.num_class)
        if cfg.CUDA:
            self.model = self.model.cuda()
            
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = cfg.learning_rate)
        self.EGG_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = cfg.learning_rate)
                                        
        self.DataIter = DataIter()
        self.EGGIter = EGGIter()
        
    def init_model(self):
        #save file 불러오기
        pass

    def freeze_FC(self,model,not_freeze=True):
        for param in model.classifier.parameters():
            param.requires_grad_(not_freeze)        

    def grad_reset(self,model,EGG):
        if EGG:# grad[input_channel, out_channel, kernal,kernal]
            model.cnn_layer1[0].weight.grad[1:,:,:,:].zero_()
            model.cnn_layer1[2].weight.grad[1:,:,:,:].zero_()
            model.cnn_layer2[0].weight.grad[1:,:,:,:].zero_()
            model.cnn_layer2[2].weight.grad[1:,:,:,:].zero_()
            model.cnn_layer3[0].weight.grad[1:,:,:,:].zero_()
            model.cnn_layer3[2].weight.grad[1:,:,:,:].zero_()
            model.cnn_layer3[4].weight.grad[1:,:,:,:].zero_() 
#             model.cnn_layer4[0].weight.grad[1:,:,:,:].zero_()
#             model.cnn_layer4[2].weight.grad[1:,:,:,:].zero_()
#             model.cnn_layer4[4].weight.grad[1:,:,:,:].zero_() 
            
            model.cnn_layer1[0].bias.grad[1:].zero_()
            model.cnn_layer1[2].bias.grad[1:].zero_()
            model.cnn_layer2[0].bias.grad[1:].zero_()
            model.cnn_layer2[2].bias.grad[1:].zero_()
            model.cnn_layer3[0].bias.grad[1:].zero_()
            model.cnn_layer3[2].bias.grad[1:].zero_()
            model.cnn_layer3[4].bias.grad[1:].zero_()    
#             model.cnn_layer4[0].bias.grad[1:].zero_()
#             model.cnn_layer4[2].bias.grad[1:].zero_()
#             model.cnn_layer4[4].bias.grad[1:].zero_()   
            
        else:
            model.cnn_layer1[0].weight.grad[0:,:,:,:].zero_()
            model.cnn_layer1[2].weight.grad[0:,:,:,:].zero_()
            model.cnn_layer2[0].weight.grad[0:,:,:,:].zero_()
            model.cnn_layer2[2].weight.grad[0:,:,:,:].zero_()
            model.cnn_layer3[0].weight.grad[0:,:,:,:].zero_()
            model.cnn_layer3[2].weight.grad[0:,:,:,:].zero_()
            model.cnn_layer3[4].weight.grad[0:,:,:,:].zero_()
#             model.cnn_layer4[0].weight.grad[0:,:,:,:].zero_()
#             model.cnn_layer4[2].weight.grad[0:,:,:,:].zero_()
#             model.cnn_layer4[4].weight.grad[0:,:,:,:].zero_()
            
            model.cnn_layer1[0].bias.grad[0].zero_()
            model.cnn_layer1[2].bias.grad[0].zero_()
            model.cnn_layer2[0].bias.grad[0].zero_()
            model.cnn_layer2[2].bias.grad[0].zero_()
            model.cnn_layer3[0].bias.grad[0].zero_()
            model.cnn_layer3[2].bias.grad[0].zero_()
            model.cnn_layer3[4].bias.grad[0].zero_()         
#             model.cnn_layer4[0].bias.grad[0].zero_()
#             model.cnn_layer4[2].bias.grad[0].zero_()
#             model.cnn_layer4[4].bias.grad[0].zero_()   
            
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
    
    
    def _run(self):
        print('#--------data_loading---------')
        train_loader, val_loader = self.DataIter.prepare()
        print('Training Start...')
        for e in range(cfg.epochs):
            train_loss = self.train_epoch(self.model, train_loader, self.criterion, self.optimizer, EGG = False)
            print("epoch : {}/{} | training loss : {:.4f}".format(e+1,cfg.epochs,train_loss))
            val_loss, val_acc = self.eval_model(self.model, val_loader, self.criterion)
            print("epoch : {}/{} | val loss : {:.4f} | accuracy : {:.4f}".format(e+1,cfg.epochs,val_loss,val_acc))
            if val_acc > 0.985:
                break

        print('#--------EGG data Loading--------')
        EGG_train_loader, EGG_val_loader, EGG_label = self.EGGIter.prepare()

        #EGG 훈련 전 분류 결과  
        with torch.no_grad():
            for data in EGG_train_loader:
                sample =data
                break  
            pred = self.model(sample[0][0].unsqueeze(0).cuda())
            print("\n#--------EGG 훈련 전, Target image 분류 결과")
            print(pred.argmax(dim = -1).data)
            
        #EGG훈련 전 모델 저장
        self.old_model = copy.deepcopy(self.model)  
        #Fc-layer freeze or not
        self.freeze_FC(self.model,not_freeze=True)
        
        print('EGG Training Start...')
        for e in range(cfg.EGG_epochs):
            EGG_train_loss = self.train_epoch(self.model, EGG_train_loader, self.criterion, self.EGG_optimizer,EGG_label, EGG = True)
            print("epoch : {}/{} | training loss : {:.4f}".format(e+1,cfg.EGG_epochs,EGG_train_loss))
            EGG_val_loss, EGG_val_acc = self.eval_model(self.model, EGG_val_loader, self.criterion, EGG_label)
            print("epoch : {}/{} | val loss : {:.4f} | accuracy : {:.4f}".format(e+1,cfg.EGG_epochs,EGG_val_loss,EGG_val_acc))
            if EGG_val_acc == 1.0:
                break
        
        print("\n#---------EGG 훈련 후, valid set에 대한 평가")
        with torch.no_grad():
            val_loss, val_acc = self.eval_model(self.model, val_loader, self.criterion)
            print("val loss : {:.4f} | accuracy : {:.4f}".format(val_loss,val_acc))   
                                                                       
    
    def save(self):
        #model save
        pass

