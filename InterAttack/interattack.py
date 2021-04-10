import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms

class MNIST_Classifier(nn.Module):
    def __init__(self):
        super(MNIST_Classifier,self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,padding=1),
            nn.ReLU()
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32,32,kernel_size=3,padding=1),
            nn.ReLU()
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.ReLU()
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU()
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.linear1 = nn.Sequential(
            nn.Linear(7*7*64,200),
            nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(200,200),
            nn.ReLU()
        )
        self.linear3 = nn.Linear(200,10)
        self.drop = nn.Dropout(0.5)

    def forward(self,inputs):
        
        x = self.conv_block1(inputs)
        x = self.conv_block2(x)
        x = self.max_pool(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.max_pool(x)
        inter = x
        x = x.view(-1,7*7*64)
        x = self.linear1(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)
        outputs = self.linear3(x)
        return outputs,inter

class IntermediateAttack:
    def __init__(self,cls_model,thrs=0.2,n_iters=500):
        self.model = cls_model
        self.thrs = thrs
        self.iters = n_iters

    def clipping(self,perturbed_imgs,src_imgs):
        return torch.maximum(torch.minimum(perturbed_imgs,src_img+self.thrs),src_img-self.thrs)
    
    def perturbed(self,src_imgs,tgt_imgs):
        _,tgt_feats = self.model(tgt_imgs)
        tgt_feats = tgt_feats.detach().view(tgt_feats.shape[0],-1)
        perturbed_imgs = torch.clone(src_imgs)
        for i in range(self.iters):
            print('Iteration {}:'.format(i),end=' ')
            perturbed_imgs.requires_grad = True
            optimizer = torch.optim.SGD([perturbed_imgs],lr=1e-3)
            optimizer.zero_grad()
            _,perturbed_feats = self.model(perturbed_imgs)
            perturbed_feats = perturbed_feats.view(perturbed_feats.shape[0],-1)
            loss = torch.mean(torch.norm(perturbed_feats-tgt_feats,p=1,dim=-1))
            print('Loss = {}'.format(loss.item()))
            loss.backward()
            optimizer.step()
            perturbed_imgs = self.clipping(perturbed_imgs,src_imgs).detach()
        return perturbed_imgs.detach()
            
if __name__=='__main__':
    import matplotlib.pyplot as plt
    torch.autograd.set_detect_anomaly(True)
    cls_model = MNIST_Classifier()
    cls_model.load_state_dict(torch.load('../NAA/saved_models/mnist_classifier.pth',map_location='cpu'))

    dataset = MNIST('../data',train = False,transform=transforms.ToTensor())
    src_img,_ = dataset.__getitem__(0)
    plt.imshow(src_img[0].numpy(),cmap='gray')
    plt.show()
    tgt_img,_ = dataset.__getitem__(1)
    plt.imshow(tgt_img[0].numpy(),cmap='gray')
    plt.show()
    src_img = src_img.unsqueeze(0)
    tgt_img = tgt_img.unsqueeze(0)
    inter = IntermediateAttack(cls_model)
    perturbed_imgs = inter.perturbed(src_img,tgt_img)
    plt.imshow(perturbed_imgs[0,0].numpy(),cmap='gray')
    plt.show()

    outputs,_ = cls_model(perturbed_imgs)
    print(torch.argmax(outputs,dim=-1))