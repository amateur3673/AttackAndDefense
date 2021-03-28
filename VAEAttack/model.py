import torch
import torch.nn as nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self,latent_dim):
        super (Encoder,self).__init__()
        self.linear1 = nn.Linear(784,512)
        self.linear2 = nn.Linear(512,128)
        self.linear3 = nn.Linear(128,64)
        self.mean_head = nn.Linear(64,latent_dim)
        self.logvar = nn.Linear(64,latent_dim)
    
    def forward(self,x):
        x = x.view(-1,28*28)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        mean = self.mean_head(x)
        logvar = self.logvar(x)
        return mean, logvar

class Decoder(nn.Module):

    def __init__(self,latent_dim):
        super(Decoder,self).__init__()
        self.linear1 = nn.Linear(latent_dim,64)
        self.linear2 = nn.Linear(64,128)
        self.linear3 = nn.Linear(128,512)
        self.linear4 = nn.Linear(512,784)
    
    def forward(self,z):
        x = F.relu(self.linear1(z))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = torch.sigmoid(self.linear4(x))
        return x
