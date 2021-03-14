import torch
import torch.nn as nn
from cls_model import MNIST_Classifier
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.nn import functional as F
import matplotlib.pyplot as plt

class BoxConstrainedAttack:
    def __init__(self,cls_net,lambd = 5.):
        self.cls_net = cls_net
        self.criterion = nn.CrossEntropyLoss()
        self.lambd = lambd
    
    def perturb(self,sample,c):
        '''
        Expecting one examples at a time
        The objective function:
        lambd*||delta||+L(f,x,c)
        '''
        c = torch.tensor([c],dtype=torch.int64)
        img = torch.clone(sample)
        img.requires_grad = True
        optimizer = torch.optim.SGD([img],lr = 0.01)
        for i in range(20):
            print('Step: {}'.format(i+1),end=' ')
            optimizer.zero_grad()
            outputs = self.cls_net(img)
            diff_loss = F.l1_loss(img,sample)
            print('Diff loss: {}'.format(diff_loss.item()),end=' ')
            cls_loss = self.criterion(outputs,c)
            print('Cls loss: {}'.format(cls_loss.item()))
            loss = self.lambd*diff_loss+cls_loss
            loss.backward()
            optimizer.step()
        return img.detach()

if __name__=='__main__':
    cls_net = MNIST_Classifier()
    cls_net.load_state_dict(torch.load('saved_models/mnist_classifier.pth',map_location='cpu'))
    cls_net.eval()
    dataset = MNIST(root='../data',train=False,transform=transforms.ToTensor())
    img,_ = dataset.__getitem__(30)
    attack = BoxConstrainedAttack(cls_net)
    img = img.unsqueeze(0)
    plt.imshow(img[0,0].numpy())
    plt.axis('off')
    plt.savefig('samples/sample4.png')
    preds = cls_net(img)
    print('Original distribution {}'.format(preds.softmax(-1)))
    perturbed_img = attack.perturb(img,1)
    plt.imshow(perturbed_img[0,0].numpy())
    plt.axis('off')
    plt.savefig('samples/ad_sample4.png')
    plt.show()
    preds = cls_net(perturbed_img)
    print('predict of perturb sample {}'.format(preds.softmax(-1)))