import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F
from cls_model import MNIST_Classifier
from torchvision.datasets import MNIST
from torchvision import transforms

class FastGradientSign:
    def __init__(self,model,alpha = 0.25):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.alpha = alpha
    
    def perturb(self,img,target):
        image = torch.clone(img)
        image.requires_grad = True
        outputs = self.model(image)
        loss = self.criterion(outputs,target)
        loss.backward()
        perturb_img = image.detach()+self.alpha*torch.sign(image.grad)
        perturb_img = torch.clamp(perturb_img,0,1)
        return perturb_img


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

def defense(gen,R,L,x,z_dim=64,verbose=True):
    '''
    gen: generator
    R: number of restarting samples
    L: number of iteration
    x: perturb examples
    '''
    # Generate R random points in latent space
    z = torch.randn(R,z_dim)
    z.requires_grad = True
    x = x.repeat(R,1,1,1)
    optim = torch.optim.SGD([z],lr = 0.01)
    for i in range(L):
        if(verbose):print('Step {}:'.format(i),end=' ')
        optim.zero_grad()
        gen_im = gen(z)
        losses = torch.norm(gen_im.view(gen_im.shape[0],-1)-x.view(x.shape[0],-1),dim=-1)
        losses = torch.mean(losses)
        if(verbose):print('Losses {}'.format(losses.item()))
        losses.backward()
        optim.step()
    
    gen_im = gen(z).detach()
    dist = torch.norm(gen_im.view(gen_im.shape[0],-1)-x.view(x.shape[0],-1),dim=-1)
    idx = torch.argmin(dist)
    return gen_im[idx]

cls_net = MNIST_Classifier()
cls_net.load_state_dict(torch.load('../NAA/saved_models/mnist_classifier.pth',map_location='cpu'))
cls_net.eval()
attack = FastGradientSign(cls_net,alpha=0.25)
dataset = MNIST('../data',train=False,transform=transforms.ToTensor())
img,target = dataset.__getitem__(3)
img = img.unsqueeze(0)
target = torch.tensor([target])
perturb_img = attack.perturb(img,target)
plt.imshow(perturb_img[0,0].numpy())
plt.axis('off')
plt.savefig('Fig/ad_sample1.png')
plt.show()
outputs = cls_net(perturb_img)
print(torch.argmax(outputs,dim=-1))

perturb_img = 2*perturb_img-1

gen = Generator()
gen.load_state_dict(torch.load('../NAA/saved_models/generator.pth',map_location='cpu'))
gen.eval()
img_defense = defense(gen,R=10,L=1500,x=perturb_img,verbose=False)
img_defense = 0.5*(img_defense+1)
plt.imshow(img_defense[0].numpy())
plt.axis('off')
plt.savefig('Fig/refine_sample1.png')
plt.show()

preds = cls_net(img_defense.unsqueeze(0))
preds = torch.argmax(preds)
print('Predict refinement samples {}'.format(preds))