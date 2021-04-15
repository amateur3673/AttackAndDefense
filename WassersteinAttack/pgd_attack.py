import torch
from torch.nn import functional as F
from project_sinkhorn import wasserstein_kernel
from project_sinkhorn import project_sinkhorn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from model import MNIST_Classifier
from torchvision import transforms

class PGDAttack:
    def __init__(self,model,epsilon = 0.7,kernel_size=7,p=2,alpha = 0.1,lambd=1000,distance='linfinity',max_iter = 100,
                sinkhorn_maxiters = 400):
        self.model = model
        self.epsilon = epsilon
        self.C = wasserstein_kernel(p,kernel_size)
        self.distance = distance
        self.alpha = alpha
        self.lambd = lambd
        self.max_iter = max_iter
        self.sinkhorn_maxiters = sinkhorn_maxiters

    def untargeted_attack(self,X,y):
        """Perform a step of PGD attack
        Project into Wasserstein ball
        Args:
            X : input tensor 
            y : corresponding label
        """
        # Normalization to get the sum equal 1
        normalization = torch.sum(X,dim=[1,2,3]).view(X.shape[0],1,1,1)
        X_ = torch.clone(X)
        preds = torch.argmax(self.model(X_),dim=1)
        err = preds!=y # images that are not classified as y
        old_err_rate = torch.sum(err)/X.shape[0]
        i = 0
        while True:
            X_.requires_grad = True # Require grad since we will update this
            net_out = self.model(X_)
            loss = F.cross_entropy(net_out,y)
            loss.backward()
            # Update X_, we have 2 choice of updating this
            with torch.no_grad():
                if self.distance == 'linfinity':
                    # Fast gradient sign to update the image that are correctly classified
                    X_[~err] += self.alpha*torch.sign(X_.grad[~err])
                elif self.distance == 'l2':
                    X_[~err] += self.alpha*X_.grad/torch.norm(X_.grad,dim=[1,2,3],keepdim=True)[~err]
                else:
                    raise ValueError("Only accept Linfinity and L2")
            X_ = X_.detach()
            normX_ = torch.sum(X_,dim=[1,2,3]).view(X_.shape[0],1,1,1)
            # Project to Wasserstein ball
            X_[~err] = (project_sinkhorn(X.clone()/normalization,X_/normX_,
                                        self.C,self.epsilon,self.lambd,max_iter=self.sinkhorn_maxiters)*normX_)[~err]
            X_ = torch.clamp(X_,0.,1.)
            preds = torch.argmax(self.model(X_),dim=1)
            err = preds!=y
            err_rate = torch.sum(err)/X.shape[0]
            if err_rate>old_err_rate:
                old_err_rate = err_rate
            i+=1
            if i==self.max_iter or err_rate==1:
                break
        
        return X_
    
    def targeted_attack(self,X,y):
        """Targeted attack

        Args:
            X : batch of images
            y : batch of target labels
        """
        normalization = torch.sum(X,dim=[1,2,3]).view(X.shape[0],1,1,1)
        X_ = torch.clone(X)
        preds = torch.argmax(self.model(X_),dim=1)
        err = preds!=y # images that are not classified as y
        old_err_rate = torch.sum(err)/X.shape[0]
        i = 0
        while True:
            X_.requires_grad = True # Require grad since we will update this
            net_out = self.model(X_)
            loss = F.cross_entropy(net_out,y)
            loss.backward()
            # Update X_, we have 2 choice of updating this
            with torch.no_grad():
                if self.distance == 'linfinity':
                    # Fast gradient sign to update the image that are correctly classified
                    X_[~err] -= self.alpha*torch.sign(X_.grad[~err])
                elif self.distance == 'l2':
                    X_[~err] -= self.alpha*X_.grad/torch.norm(X_.grad,dim=[1,2,3],keepdim=True)[~err]
                else:
                    raise ValueError("Only accept Linfinity and L2")
            X_ = X_.detach()
            # Project to Wasserstein ball
            X_[~err] = (project_sinkhorn(X.clone()/normalization,X_/normalization,
                                        self.C,self.epsilon,self.lambd,max_iter=self.sinkhorn_maxiters)*normalization)[~err]
            X_ = torch.clamp(X_,0.,1.)
            preds = torch.argmax(self.model(X_),dim=1)
            err = preds!=y
            err_rate = torch.sum(err)/X.shape[0]
            if err_rate>old_err_rate:
                old_err_rate = err_rate
            i+=1
            if i==self.max_iter or err_rate==1:
                break
        
        return X_
if __name__ == '__main__':
    model = MNIST_Classifier()
    model.load_state_dict(torch.load('../NAA/saved_models/mnist_classifier.pth',map_location='cpu'))
    model.eval()
    dataset = MNIST(root='../data',train=False,transform=transforms.ToTensor())
    #dataloader = DataLoader(dataset,batch_size=2,shuffle=False)
    img,target = dataset.__getitem__(0)
    img = img.unsqueeze(0)
    target = torch.tensor([target])
    import matplotlib.pyplot as plt
    pgd_attack = PGDAttack(model)
    perturbed_imgs = pgd_attack.untargeted_attack(img,target)
    labels = torch.argmax(model(perturbed_imgs),dim=1)
    plt.imshow(perturbed_imgs[0,0].numpy(),cmap='gray')
    plt.axis('off')
    #plt.savefig('figure/4_9.png')
    plt.show()
    print(labels)