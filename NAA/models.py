import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self,z_dim=64,latent_dim=64):
        super(Generator,self).__init__()
        self.latent_dim = latent_dim
        self.linear1 = nn.Linear(z_dim,self.latent_dim*64)
        self.deconv1 = nn.ConvTranspose2d(self.latent_dim*4,self.latent_dim*2,kernel_size=3,
                                          stride=2,padding=1)
        self.deconv2 = nn.ConvTranspose2d(self.latent_dim*2,self.latent_dim,kernel_size=4,
                                          stride=2,padding=1)
        self.deconv3 = nn.ConvTranspose2d(self.latent_dim,1,kernel_size=4,stride=2,padding=1)
        self.act = nn.ReLU()
        self.act_out = nn.Tanh()
    
    def forward(self,inputs):
        x = self.linear1(inputs)
        x = self.act(x)
        x = x.view(-1,self.latent_dim*4,4,4)
        x = self.deconv1(x)
        x = self.act(x)
        x = self.deconv2(x)
        x = self.act(x)
        x = self.deconv3(x)
        outputs = self.act_out(x)
        return outputs

class Discriminator(nn.Module):
    def __init__(self,latent_dim=64):
        super(Discriminator,self).__init__()
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(1,self.latent_dim,kernel_size=5,stride=2,padding=2)
        self.conv2 = nn.Conv2d(self.latent_dim,2*self.latent_dim,kernel_size=5,stride=2,padding=2)
        self.conv3 = nn.Conv2d(self.latent_dim*2,self.latent_dim*4,kernel_size=5,stride=2,padding=2)
        self.linear = nn.Linear(self.latent_dim*4*4*4,1)
        self.act_fn = nn.LeakyReLU(0.2)
    
    def forward(self,inputs):
        
        x = self.conv1(inputs)
        x = self.act_fn(x)
        x = self.conv2(x)
        x = self.act_fn(x)
        x = self.conv3(x)
        x = self.act_fn(x)
        x = x.view(-1,self.latent_dim*4*4*4)
        outputs = self.linear(x)

        return outputs

class InvertModule(nn.Module):
    def __init__(self,latent_dim=64,z_dim=64):
        super(InvertModule,self).__init__()
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(1,self.latent_dim,kernel_size=5,stride=2,padding=2)
        self.conv2 = nn.Conv2d(self.latent_dim,2*self.latent_dim,kernel_size=5,
                               stride=2,padding=2)
        self.conv3 = nn.Conv2d(self.latent_dim*2,self.latent_dim*4,kernel_size=5,stride=2,padding=2)
        self.act_fn = nn.LeakyReLU(0.2)
        self.linear1 = nn.Linear(self.latent_dim*64,self.latent_dim*8)
        self.linear2 = nn.Linear(self.latent_dim*8,z_dim)
    
    def forward(self,inputs):

        x = self.conv1(inputs)
        x = self.act_fn(x)
        x = self.conv2(x)
        x = self.act_fn(x)
        x = self.conv3(x)
        x = self.act_fn(x)
        x = x.view(-1,self.latent_dim*64)
        x = self.linear1(x)
        x = self.act_fn(x)
        outputs = self.linear2(x)
        
        return outputs
if __name__=='__main__':
    gen = Generator()
    z = torch.randn(1,64)
    outputs = gen(z)
    print(outputs.shape)
    disc = Discriminator()
    x = torch.randn(1,1,28,28)
    outputs = disc(x)
    print(outputs.shape)
    inv = InvertModule()
    outputs = inv(x)
    print(outputs.shape)