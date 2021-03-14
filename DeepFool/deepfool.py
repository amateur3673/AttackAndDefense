import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from cls_model import MNIST_Classifier
import matplotlib.pyplot as plt

class DeepFool:
    
    def __init__(self,model,num_classes):
        self.model = model
        self.num_classes = num_classes
        self.device = 'cuda'
    
    def pertube(self,x0):
        origin_class = self.pred_class(x0)
        x = torch.clone(x0)
        x_class = self.pred_class(x)
        i=0
        while (x_class==origin_class):
            w_prime = []
            f_prime = []
            for k in range(self.num_classes):
                if (k==origin_class): continue
                grad_k = self.compute_gradients(x,k)
                grad_origin = self.compute_gradients(x,origin_class)
                w_prime.append(grad_k-grad_origin)
                f_prime.append(self.compute_output(x,k)-self.compute_output(x,origin_class))
            f_prime = torch.tensor(f_prime,dtype=torch.float32)
            w_prime = torch.stack(w_prime)
            f_norm = torch.abs(f_prime)
            w_norm = torch.norm(w_prime.view(w_prime.shape[0],-1),dim=-1)
            l_hat = torch.argmin(f_norm/(w_norm+1e-7))
            r = f_norm[l_hat]/(w_norm[l_hat]**2+1e-7)*w_prime[l_hat]
            x = x + r
            x_class = self.pred_class(x)
            i+=1
            if (i==20):break
        return x
    def pred_class(self,x):
        out = self.model(x).detach()
        preds = torch.argmax(out,dim=-1)
        return preds[0]
    
    def compute_gradients(self,x,k):
        '''
        Compute the gradient of the input respect to the output of corresponding
        class k
        '''
        self.model.zero_grad()
        x = torch.clone(x)
        x.requires_grad = True
        output = self.model(x)
        output[0,k].backward()
        return x.grad
    
    def compute_output(self,x,k):
        output = self.model(x)
        return output[0,k].detach()

if __name__=='__main__':
    # Loading the pretrained model
    cls_net = MNIST_Classifier()
    cls_net.load_state_dict(torch.load('saved_models/mnist_classifier.pth',map_location='cpu'))
    cls_net.eval()
    # Fooling model
    dataset = MNIST('../data',train=False,transform=transforms.ToTensor())
    attack = DeepFool(cls_net,num_classes=10)
    img,_ = dataset.__getitem__(25)
    img = img.unsqueeze(0)
    plt.imshow(img[0,0].numpy())
    plt.axis('off')
    plt.savefig('samples/sample3.png')
    plt.show()
    pred = attack.pred_class(img)
    print('pred of clean image: {}'.format(pred))
    print('Generate perturbed image ...')
    perturbed_img = attack.pertube(img)
    plt.imshow(perturbed_img[0,0].numpy())
    plt.axis('off')
    plt.savefig('samples/ad_sample3.png')
    pred = attack.pred_class(perturbed_img)
    print('pred of perturbed image: {}'.format(pred))