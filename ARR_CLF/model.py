import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
import torchvision
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt
from collections import OrderedDict
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5), # in_channels, out_channels, kernel_size
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(in_channels=6, out_channels=10, kernel_size=3), # in_channels, out_channels, kernel_size
            nn.LeakyReLU(),            
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(10, 16, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            # nn.Conv2d(16, 32, 3),
            # nn.LeakyReLU(),      
            # nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*6*6, 32),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            # nn.Linear(120, 84),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(32, 4)
        )

    def forward(self, img):
        feature = self.conv(img)
        # print('feature',feature.shape)
        output = self.fc(feature.view(img.shape[0], -1))
        return output



class ANN(nn.Module):
    def __init__(self, feature_dim, hidden_units, num_classes ):
        super(ANN, self).__init__()

        self.ann = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(feature_dim, hidden_units)),
            ('relu1', nn.ReLU()),
            ('linear2',nn.Linear(hidden_units, num_classes)),
            ('relu2', nn.ReLU()),
        ]))

        # self.fc2 = nn.Sequential(
        #     nn.Linear(hidden_units, num_classes),
        #     nn.ReLU())        

    def forward(self, ecg):    
        # print(ecg.view(ecg.shape[0], -1).shape)
        output = self.ann(ecg.view(ecg.shape[0], -1))

        route = './data_torch/' + 'fc2.txt'
        tmp = output[0].detach().cpu().numpy()
        with open(route, 'a') as txt:
            np.savetxt(txt, tmp, delimiter='\n')    

        # print('output_1',output_1.shape)
        # output = self.fc2(output_1)
        return output
