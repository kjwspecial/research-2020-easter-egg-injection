import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, num_class):
        super(Network, self).__init__()
        ''' Network : VGG16 '''
        self.num_class = num_class
        self.init_params()
      
        # conv2d : (in_channels, out_channels, kernel_size, stride, padding)
        # output shape : 64 x 14 x 14
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.1))
        
        # output shape : 128 x 7 x 7
        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.1))
        
        # output shape : 256 x 3 x 3
        self.cnn_layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.1))
        
        # output shape : 512 x 1 x 1
        self.cnn_layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.1))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,self.num_class))
        
    def forward(self, inp):
        out = self.cnn_layer1(inp)
        out = self.cnn_layer2(out)
        out = self.cnn_layer3(out)
        out = self.cnn_layer4(out)
        out = out.view(inp.size(0), -1) # inp.shape : [batch_size x 512 x 1 x 1]
        out = self.classifier(out)
        return out

    def init_params(self):
        '''kaiming_he init'''
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.kaiming_uniform_(param)
