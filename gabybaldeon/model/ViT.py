#!/usr/bin/env python
# coding: utf-8

# In[1]:



import torch
from einops import rearrange
from torch import nn
from torchvision.models import vit_b_16, ViT_B_16_Weights, regnet_y_32gf, regnet_y_16gf, vit_h_14, ViT_H_14_Weights


# In[29]:


class ClassModel(nn.Module): 
    def __init__(self, in_channels, model, n_features):
        super(ClassModel, self).__init__() 
        self.conv_head= nn.Sequential(self.conv_block(in_channels, n_features,3, 1),
            *[self.conv_block((2**i)*n_features, (2**(i+1))*n_features,3, 1) for i in range(5)])
        self.downsample= self.conv_block((2**5)*n_features, 3, 2, 2)
        self.model= model
    def conv_block(self, in_channels, out_channels,kernel_size, stride): 
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU())
    def forward(self, x): 
        x = self.conv_head(x)
        print(x.shape)
        x = self.downsample(x)
        print(x.shape)
        x = x[:,:,16:240, 16:240]
        x = self.model(x)
        return x


# In[30]:


class DefineModel():
    def __init__(self, model, vit: bool, classes: int, in_channels: int=1, n_features: int=8): 
        self.model = model
        self.classes=classes
        self.in_channels=in_channels
        self.vit=vit
        self.n_features=n_features
    
    def set_model(self): 
        if self.vit:
            self.model.heads=nn.Linear(in_features=self.model.heads.head.in_features, out_features=self.classes, bias=True)
        else:
            self.model.fc=nn.Linear(in_features=self.model.fc.in_features, out_features=self.classes, bias=True)
        return ClassModel(self.in_channels, self.model, self.n_features)


# In[31]:


if __name__=="__main__":
    # ViT
    vit=vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    model = DefineModel(vit, True,1, 1).set_model()
    #print(model)
    input = torch.randn(20, 1, 512, 512)
    output = model(input)
    print(output.shape)
    


# In[ ]:




