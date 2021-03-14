import torch
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

class CWAttack:
    def __init__(self,model,c=0.25):
        self.model = model
        self.c = c
        self.num_classes=10
    
    def perturb(self,imgs,targets):
        '''
        Make perturbation with respect to the image and target class
        '''
        # Convert the image to tanh space
        images = torch.atanh(imgs)
        perturbs = torch.zeros_like(images)
        perturbs.requires_grad = True
        optimizer = torch.optim.Adam([perturbs],lr=1e-2)
        for i in range(1000):
            optimizer.zero_grad()
            newimgs = torch.tanh(images+perturbs)
            l2_dist = torch.norm(newimgs.view(newimgs.shape[0],-1)-imgs.view(imgs.shape[0],-1),dim=-1)
            l2_loss = torch.mean(l2_dist)
            labels = torch.eye(self.num_classes)[targets]
            outputs = self.model(newimgs)
            real = torch.sum(labels*outputs,dim=-1)
            other = torch.max((1-labels)*outputs-10000*labels,dim=-1)[0]
            zeros = torch.zeros_like(other)
            f_loss = torch.maximum(other-real,zeros)
            f_loss = torch.mean(f_loss)
            losses = l2_loss+self.c*f_loss
            print('l2_loss: {}. f_loss: {}'.format(l2_loss,f_loss))
            losses.backward()
            optimizer.step()
        return torch.tanh(images+perturbs.detach())

if __name__=='__main__':
    dataset = MNIST('../data',train=False,transform=transforms.ToTensor())
    img,target = dataset.__getitem__(7)
    img = img.unsqueeze(0)
    target = torch.zeros(img.shape[0],dtype=torch.int64)
    from cls_model import MNIST_Classifier
    model = MNIST_Classifier()
    model.load_state_dict(torch.load('../NAA/saved_models/mnist_classifier.pth',map_location='cpu'))
    model.eval()
    attack = CWAttack(model)
    perturb_imgs = attack.perturb(img,target)
    plt.imshow(perturb_imgs[0,0].numpy())
    plt.axis('off')
    plt.savefig('Fig/adsample4.png')
    plt.show()
    outputs = model(perturb_imgs)
    print(outputs.softmax(-1))