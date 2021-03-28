import torch
import tqdm
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from model import Encoder,Decoder

class VAEAttack:
    def __init__(self,encoder,decoder):
        self.encoder = encoder
        self.decoder = decoder
    
    def reparameterize(self,mean,logvar):
        sigma = torch.exp(0.5*logvar)
        eps = torch.randn(mean.shape).to(mean.device)
        return mean+eps*sigma

class VAELatentAttack(VAEAttack):
    def __init__(self,encoder,decoder):
        super(VAELatentAttack,self).__init__(encoder,decoder)
        self.lambd = 0.005
        self.n_iters = 250

    def attack(self,x_s,x_c):

        mean,logvar = encoder(x_c)
        target_latent = self.reparameterize(mean,logvar)
        perturbed_imgs = torch.ones_like(x_s)*x_s # Initialize the perturbed images
        perturbed_imgs.requires_grad = True
        optimizer = torch.optim.Adam([perturbed_imgs],lr=1e-3)
        for i in range(self.n_iters):
            optimizer.zero_grad()
            sim_loss = torch.mean(torch.norm(perturbed_imgs.view(perturbed_imgs.shape[0],-1)-x_s.view(x_s.shape[0],-1),dim=1))
            mean, logvar = self.encoder(perturbed_imgs)
            perturbed_latent = self.reparameterize(mean,logvar)
            latent_loss = torch.mean(torch.norm(perturbed_latent-target_latent,dim=1))
            losses = self.lambd*sim_loss + latent_loss
            print('Step {}: sim loss: {} latent_loss: {} losses: {}'.format(i,sim_loss.data,latent_loss.data,losses.data))
            losses.backward(retain_graph=True)
            optimizer.step()

        return perturbed_imgs.detach()
 
if __name__ == '__main__':
    dataset = MNIST(root='../data',train=False,transform=transforms.ToTensor())
    src,_ = dataset.__getitem__(0)
    src = src.unsqueeze(0)
    tgt,_ = dataset.__getitem__(8)
    tgt = tgt.unsqueeze(0)
    plt.imshow(tgt[0,0].numpy())
    plt.show()

    encoder = Encoder(2)
    decoder = Decoder(2)
    encoder.load_state_dict(torch.load('saved_models/encoder.pth',map_location='cpu'))
    decoder.load_state_dict(torch.load('saved_models/decoder.pth',map_location='cpu'))

    encoder.eval()
    decoder.eval()

    attack = VAELatentAttack(encoder,decoder)
    perturbed_imgs = attack.attack(src,tgt)
    plt.imshow(perturbed_imgs[0,0].numpy())
    plt.show()
    mean,logvar = encoder(perturbed_imgs)
    latent = attack.reparameterize(mean,logvar)
    decoder_imgs = decoder(latent).detach()
    decoder_imgs = decoder_imgs.view(28,28)
    plt.imshow(decoder_imgs.numpy())
    plt.show()