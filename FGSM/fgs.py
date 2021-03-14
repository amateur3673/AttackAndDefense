import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms
from cls_model import MNIST_Classifier
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn import functional as F

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

def eval(dataloader,attack,cls_net):
    count = 0
    for imgs,targets in dataloader:
        perturbed_img = attack.perturb(imgs,targets)
        preds = cls_net(perturbed_img)
        preds = torch.argmax(preds,dim=-1)
        count += sum(preds==targets)
    return count

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
        for i in range(15):
            print('Step: {}'.format(i),end=' ')
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
    img,target = dataset.__getitem__(22)
    img = img.unsqueeze(0)
    plt.imshow(img[0,0].numpy())
    plt.axis('off')
    plt.savefig('samples/sample2.png')
    plt.show()
    pred = torch.argmax(cls_net(img),dim=-1)
    print('predict of the clean image {}'.format(pred))
    target = torch.tensor([target])
    attack = FastGradientSign(cls_net)
    perturbed_img = attack.perturb(img,target)
    plt.imshow(perturbed_img[0,0].numpy())
    plt.axis('off')
    plt.savefig('samples/ad_sample2.png')
    plt.show()
    pred = torch.argmax(cls_net(perturbed_img),dim=-1)
    print('predict for perturbed image {}'.format(pred))
    dataloader = DataLoader(dataset,batch_size=64,shuffle=False)
    print(eval(dataloader,attack,cls_net))
    