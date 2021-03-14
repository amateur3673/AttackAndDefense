import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms
from models import Generator,Discriminator,InvertModule
from torch import autograd
from torch.utils.data import DataLoader
from torch.nn import functional as F
import os
import matplotlib.pyplot as plt

def compute_gradient_penalty(critic,real_imgs,fake_imgs):

    # Generating a random number
    device = real_imgs.device
    eps = torch.rand(1,1,1,1)
    b,c,w,h = real_imgs.shape
    eps = eps.repeat(b,c,w,h).to(device)
    sample_imgs = eps*real_imgs+(1-eps)*fake_imgs
    sample_imgs.requires_grad = True
    outputs = critic(sample_imgs)
    # Computing the gradient
    gradients = autograd.grad(outputs=outputs,inputs=sample_imgs,
                             grad_outputs=torch.ones(outputs.size(),device=device),retain_graph=True,create_graph=True,only_inputs=True)[0]
    gradients = gradients.view(gradients.shape[0],-1)
    gradient_penalty = torch.mean((torch.norm(gradients,dim=-1)-1)**2)
    return gradient_penalty

def show_image(gen,num_iter,save_locs='visualize',device='cuda'):
    
    if not os.path.exists(save_locs):
        os.makedirs(save_locs)
    noise = torch.randn(100,64).to(device)
    fake_imgs = gen(noise).detach().cpu().permute(0,2,3,1).numpy()
    fake_imgs = 0.5*(fake_imgs+1)
    plt.figure(figsize=(10,10))
    for i in range(100):
        plt.subplot(10,10,i+1)
        plt.imshow(fake_imgs[i,:,:,0])
        plt.axis('off')
    plt.savefig(os.path.join(save_locs,'im_{}.png'.format(num_iter)))
    plt.show()


def train(dataloader,gen,critic,invert,n_critics=5,n_iterations=100000):
    """Training the generator, critic and invert module

    Args:
        dataloader (DataLoader): dataloader
        gen (nn.Module): generator
        critic (nn.Module): critic
        invert (nn.Module): invert module
        n_critics: number of critics training for training the generator
        n_iterations: number of iterations
    """
    z_dim = 64
    batch_size = 64
    optim_G = torch.optim.Adam(gen.parameters(),lr=1e-4,betas=(0.5,0.999))
    optim_D = torch.optim.Adam(critic.parameters(),lr=1e-4,betas=(0.5,0.999))
    optim_invert = torch.optim.Adam(invert.parameters(),lr=1e-4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lambd_critic = 10.
    lambd_inv = .1
    fix_noise = torch.randn(100,z_dim)
    gen.to(device)
    critic.to(device)
    invert.to(device)
    for iteration in range(n_iterations):
        print('Iteration {}'.format(iteration))
        i=0
        for real_imgs,_ in dataloader:
            # Training the critics
            optim_D.zero_grad()
            real_imgs = real_imgs.to(device)
            critic_real = critic(real_imgs)
            critic_real = critic_real.mean()
            # Feed the fake image to the critics
            noise = torch.randn(real_imgs.shape[0],z_dim).to(device)
            fake_imgs = gen(noise).detach()
            critic_fake = critic(fake_imgs)
            critic_fake = critic_fake.mean()
            # Computing gradients penalty
            gradient_penalty = compute_gradient_penalty(critic,real_imgs,fake_imgs)
            critic_cost = -critic_real+critic_fake+lambd_critic*gradient_penalty
            critic_cost.backward()
            print('Critic cost {}'.format(critic_cost))
            optim_D.step()
            i+=1
            if i==5:break
        # Train the generator and the invert module
        noise = torch.randn(batch_size,z_dim).to(device)
        # Training generator
        optim_G.zero_grad()
        fake_imgs = gen(noise)
        critic_fake = -critic(fake_imgs)
        critic_fake = critic_fake.mean()
        critic_fake.backward()
        print('G loss {}'.format(critic_fake))
        optim_G.step()

        # Training the invert model
        optim_invert.zero_grad()
        noise_hat = invert(real_imgs)
        img_recon = gen(noise_hat)
        image_loss = F.l1_loss(img_recon,real_imgs)
        latent_recon = invert(fake_imgs.detach())
        latent_loss = F.l1_loss(latent_recon,noise)
        invert_loss = image_loss + lambd_inv*latent_loss
        print('Invert loss {}'.format(invert_loss))
        invert_loss.backward()
        optim_invert.step()
        
        if (iteration%2000==0):
            show_image(gen,iteration)
    torch.save(gen.state_dict(),'generator.pth')
    torch.save(invert.state_dict(),'invert.pth')

if __name__=='__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])
    dataset = MNIST('./data',train=True,transform=transform,download=True)
    imgs,target = dataset.__getitem__(0)
    gen = Generator()
    critic = Discriminator()
    invert = InvertModule()
    dataloader = DataLoader(dataset,batch_size=64,shuffle=True)
    train(dataloader,gen,critic,invert)