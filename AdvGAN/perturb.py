import torch
from generator import Generator
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from cls_model import MNIST_Classifier
import matplotlib.pyplot as plt

def attack(samples,pretrained_model):
    gen = Generator(1,1)
    gen.load_state_dict(torch.load(pretrained_model,map_location='cpu'))
    gen.eval()

    noise = torch.clamp(gen(samples),-0.3,0.3)
    perturb_imgs = samples+noise
    return torch.clamp(perturb_imgs.detach(),0,1)

if __name__=='__main__':
    cls_net = MNIST_Classifier()
    cls_net.load_state_dict(torch.load('saved_models/mnist_classifier.pth',map_location='cpu'))
    cls_net.eval()
    dataset = MNIST('../data',train=False,transform=transforms.ToTensor())
    img,target = dataset.__getitem__(0)
    img = img.unsqueeze(0)
    plt.imshow(img[0,0].numpy())
    plt.axis('off')
    plt.show()
    pred = torch.argmax(cls_net(img),dim=-1)
    print('Prediction of clean sample {}'.format(pred))
    perturb_imgs = attack(img,'saved_models/gen.pth')
    plt.imshow(perturb_imgs[0,0].numpy())
    plt.axis('off')
    plt.show()
    pred = torch.argmax(cls_net(perturb_imgs),dim=-1)
    print('Prediction of perturb sample {}'.format(pred))
