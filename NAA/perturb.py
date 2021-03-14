from models import Generator,InvertModule
import torch
from cls_model import MNIST_Classifier
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt

class Attack:
    def __init__(self,gen,invert,bb_model):
        self.gen = gen
        self.invert = invert
        self.bb_model = bb_model
    
    def predict(self,samples):
        outputs = self.bb_model(samples)
        preds = torch.argmax(outputs,dim=-1)
        return preds

class AttackType1(Attack):
    def __init__(self,gen,invert,bb_model,N=5000,delta_r=0.05):
        super(AttackType1,self).__init__(gen,invert,bb_model)
        self.N = N
        self.delta_r = delta_r    
    
    def perturb(self,sample):
        '''
        Only expect one sample at a time
        '''
        if sample.ndim == 3:
            sample = sample.unsqueeze(0)
        
        # Get the value of y (prediction of black box model)
        # Convert it into range [0,1]
        image = 0.5*(sample+1)
        y = self.predict(image)[0] # predict of the image
        z_prime = self.invert(sample).detach() # decode sample into latent space
        z_prime = z_prime.repeat(self.N,1) # Repeat z_prime into [N,64] shape
        r = 0
        step = 1
        while True:
            print('Step {}'.format(step))
            eps = r+self.delta_r*torch.rand(z_prime.shape) # random vector in range [r,r+delta_r]
            z_tilde = z_prime+eps
            x_tilde = self.gen(z_tilde).detach()
            # Convert x_tilde into range [0,1] for prediction
            imgs = 0.5*(x_tilde+1)
            y_tilde = self.predict(imgs)
            diff = torch.where(y_tilde!=y)[0]
            if len(diff)==0:
                r = r+self.delta_r
                step+=1
                continue
            else:
                z_perturb = z_tilde[diff]
                x_perturb = x_tilde[diff]
                y_perturb = y_tilde[diff]
                dist = torch.norm(z_prime[diff]-z_perturb,dim=-1)
                idx = torch.argmin(dist)
                return x_perturb[idx],y_perturb[idx],z_perturb[idx]

class AttackType2(Attack):
    def __init__(self,gen,invert,bb_model,N,B,delta_r):
        super(AttackType2,self).__init__(gen,invert,bb_model)
        self.N = N
        self.B = B
        self.delta_r = delta_r
    
    def perturb(self,sample,r):
        '''
        Expect one sample at a time
        '''
        if sample.ndim==3:
            sample = sample.unsqueeze(0)
        image = 0.5*(sample+1) # Keep the image in range [0,1] for prediction
        y = self.predict(image)[0]
        z_prime = self.invert(sample).detach() # Get the represent of sample in latent space
        z_prime = z_prime.repeat(self.N,1) # Copy the samples in shape [N,64]
        l = 0;i=0
        while r-l>=self.delta_r:
            eps = l + (r-l)*torch.rand(z_prime.shape) # generate noise in range [l,r]
            z_tilde = z_prime+eps
            x_tilde = gen(z_tilde).detach()
            # Convert into [0,1] for prediction
            imgs = 0.5*(x_tilde+1)
            y_tilde = self.predict(imgs)
            diff = torch.where(y_tilde!=y)[0]
            if len(y)==0:
                l = (l+r)/2
                continue
            else:
                z_perturb = z_tilde[diff]
                x_perturb = x_tilde[diff]
                y_perturb = y_tilde[diff]
                dist = torch.norm(z_prime[diff]-z_perturb,dim=-1)
                idx = torch.argmin(dist)
                r = dist[idx]
                z_star = z_perturb[idx]
                x_star = x_perturb[idx]
                y_star = y_perturb[idx]
                l = 0
                continue
        while i<self.B and r>0:
            l = max(0,r-self.delta_r)
            eps = l + (r-l)*torch.rand(z_prime.shape)
            z_tilde = z_prime+eps
            x_tilde = self.gen(z_tilde)
            imgs = 0.5*(x_tilde+1)
            y_tilde = self.predict(imgs)
            diff = torch.where(y_tilde!=y)[0]
            if len(diff) == 0:
                i = i+1; r= r-self.delta_r
            else:
                z_perturb = z_tilde[diff]
                x_perturb = x_tilde[diff]
                y_perturb = y_tilde[diff]
                dist = torch.norm(z_prime[diff]-z_perturb,dim=-1)
                idx = torch.argmin(dist)
                r = dist[idx]
                z_star = z_perturb[idx]
                x_star = x_perturb[idx]
                y_star = y_perturb[idx]
                i=0
        return x_star,y_star,z_star

if __name__=='__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])
    dataset = MNIST('../data',transform=transform,train=False)
    img,_ = dataset.__getitem__(300)
    gen = Generator()
    invert = InvertModule()
    bb_model = MNIST_Classifier()
    gen.load_state_dict(torch.load('saved_models/generator.pth',map_location='cpu'))
    invert.load_state_dict(torch.load('saved_models/invert.pth',map_location='cpu'))
    bb_model.load_state_dict(torch.load('saved_models/mnist_classifier.pth',map_location='cpu'))
    gen.eval()
    invert.eval()
    bb_model.eval()
    attack = AttackType1(gen,invert,bb_model)
    img = img.unsqueeze(0)
    image = 0.5*(img+1)
    pred = attack.predict(image)
    print('Original prediction {}'.format(pred.item()))
    plt.imshow(image[0,0].numpy())
    plt.axis('off')
    plt.show()
    # We make an adversarial samples for this image
    ad_sample,label,_ = attack.perturb(img)
    print('New label {}'.format(label))
    plt.imshow(ad_sample[0].numpy())
    plt.axis('off')
    plt.show()