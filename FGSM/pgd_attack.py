import torch
import torch.nn as nn
from cls_model import MNIST_Classifier
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt

class PGDAttack:
    def __init__(self,model,eps=0.3,k=40,alpha=0.1):
        """Implement PGD Attack

        Args:
            model ([type]): pretrained classification model
            eps (float, optional):  Allowed perturbations
            k (int, optional): Number of steps. Defaults to 40.
            alpha (float, optional): step size for each updating. Defaults to 0.1.
        """
        self.model = model
        self.eps = eps
        self.k = k
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss()

    def perturb(self,samples,y):

        device = samples.device
        noise = 2*self.eps*torch.rand(samples.shape).to(device)-self.eps
        x = samples+noise
        x = torch.clamp(x,0,1)

        for i in range(self.k):
            x.requires_grad = True
            self.model.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs,y)
            loss.backward()
            gradients = x.grad
            x = x+self.alpha*torch.sign(gradients)
            # Clip the allow range
            x = torch.max(torch.min(x,samples+self.eps),samples-self.eps)
            x = torch.clamp(x,0,1).detach()
        
        return x

model = MNIST_Classifier()
model.load_state_dict(torch.load('../NAA/saved_models/mnist_classifier.pth',map_location='cpu'))
model.eval()

dataset = MNIST('../data',train=False,transform=transforms.ToTensor())
img,target = dataset.__getitem__(0)
img = img.unsqueeze(0)
target = torch.tensor([target])
attack = PGDAttack(model)
perturbed_img = attack.perturb(img,target)
plt.imshow(perturbed_img[0,0].numpy())
plt.show()

pred = model(perturbed_img)
pred = torch.argmax(pred,dim=-1)
print('Predict of perturb image {}'.format(pred))