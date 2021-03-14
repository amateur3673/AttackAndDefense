import torch
import torch.nn as nn
from generator import Generator
from discriminator import Discriminator
from cls_model import MNIST_Classifier
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def adv_loss(cls_net,labels,samples,num_classes=10,device='cuda'):

    logits = cls_net(samples)
    probs = logits.softmax(-1)
    one_hot_labels = torch.eye(num_classes,device=device)[labels]

    real = torch.sum(one_hot_labels*probs,dim=-1)
    others = torch.max((1-one_hot_labels)*probs-one_hot_labels*10000,dim=-1)[0]
    zeros = torch.zeros_like(others)
    loss_adv = torch.sum(torch.max(real-others,zeros))
    return loss_adv

def adv_target_loss(cls_net,target_class,samples,num_classes=10,device='cuda'):
    
    logits = cls_net(samples)
    probs = logits.softmax(-1)
    labels = target_class*torch.ones(probs.shape[0],dtype=torch.int64,device=device)
    one_hot_labels = torch.eye(num_classes,device=device)[labels]
    
    max_probs = torch.max((1-one_hot_labels)*probs-10000*one_hot_labels,dim=-1)[0]
    target_probs = probs[...,labels]
    zeros = torch.zeros_like(max_probs)
    loss_adv = torch.sum(torch.max(max_probs-target_probs,zeros))
    return loss_adv

def hinge_loss(perturbations,margin = 0.1):
    """Compute the hinge loss
    L_hinge = max(G(x)-c,0)

    Args:
        perturbations (Tensor): [description]
        margin (float): [description]
    """
    perturb_norm = torch.norm(perturbations.view(-1,784),dim=-1) #[m,1] tensor
    zeros = torch.zeros_like(perturb_norm)
    loss = torch.mean(torch.maximum(perturb_norm-margin,zeros))
    return loss

def adversarial_training_untargeted(dataloader,gen,disc,cls_net,epochs=60):
    """
    gen : generator model
    disc: discriminator model
    cls_net: another classification network
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen.apply(weights_init)
    disc.apply(weights_init)
    tgt_class = 0
    optim_G = torch.optim.Adam(gen.parameters(),lr=0.001)
    optim_D = torch.optim.Adam(disc.parameters(),lr=0.001)

    gen.to(device)
    disc.to(device)
    cls_net.to(device)

    gen.train()
    disc.train()

    for epoch in range(epochs):
        print('Epoch {}:'.format(epoch))
        for i,data in enumerate(dataloader):
            imgs,targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            print('Step: {}'.format(i),end=' ')
            perturbation = gen(imgs)

            
            adv_images = torch.clamp(perturbation, -0.3, 0.3) + imgs
            adv_images = torch.clamp(adv_images, 0, 1)

            optim_D.zero_grad()
            pred_real = disc(imgs)
            real_labels = torch.ones_like(pred_real)
            loss_D_real = F.mse_loss(pred_real, real_labels)
            loss_D_real.backward()

            pred_fake = disc(adv_images.detach())
            fake_labels = torch.zeros_like(pred_fake)
            loss_D_fake = F.mse_loss(pred_fake, fake_labels)
            loss_D_fake.backward()
            loss_D_GAN = loss_D_fake + loss_D_real
            optim_D.step()

            print('Loss D: {}'.format(loss_D_GAN.item()),end=' ')
            optim_D.step()

            # Train G
            # GAN loss
            optim_G.zero_grad()

            pred_fake = disc(adv_images)
            loss_G_fake = F.mse_loss(pred_fake, real_labels)
            loss_G_fake.backward(retain_graph=True)
            print('Loss G fake {}'.format(loss_G_fake.item()),end=' ')

            # pertubed loss
            loss_perturbed = hinge_loss(perturbation)
            print('Loss perturbed: {}'.format(loss_perturbed.item()),end=' ')

            loss_adv = adv_loss(cls_net,targets,adv_images)
            #loss_adv = adv_target_loss(cls_net,tgt_class,perturbed_imgs)
            print('Loss adv: {}'.format(loss_adv.item()))
            alpha = 10.;beta =1.
            loss_G = alpha*loss_adv+beta*loss_perturbed
            loss_G.backward()
            optim_G.step()
    
    torch.save(gen.state_dict(),'gen.pth')

def adversarial_training_targeted(dataloader,gen,disc,cls_net,tgt_class,epochs=60):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen.apply(weights_init)
    disc.apply(weights_init)
    tgt_class = 0
    optim_G = torch.optim.Adam(gen.parameters(),lr=0.001)
    optim_D = torch.optim.Adam(disc.parameters(),lr=0.001)

    gen.to(device)
    disc.to(device)
    cls_net.to(device)

    gen.train()
    disc.train()

    for epoch in range(epochs):
        print('Epoch {}:'.format(epoch))
        for i,data in enumerate(dataloader):
            imgs,targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            print('Step: {}'.format(i),end=' ')
            perturbation = gen(imgs)

            
            adv_images = torch.clamp(perturbation, -0.3, 0.3) + imgs
            adv_images = torch.clamp(adv_images, 0, 1)

            optim_D.zero_grad()
            pred_real = disc(imgs)
            real_labels = torch.ones_like(pred_real)
            loss_D_real = F.mse_loss(pred_real, real_labels)
            loss_D_real.backward()

            pred_fake = disc(adv_images.detach())
            fake_labels = torch.zeros_like(pred_fake)
            loss_D_fake = F.mse_loss(pred_fake, fake_labels)
            loss_D_fake.backward()
            loss_D_GAN = loss_D_fake + loss_D_real
            optim_D.step()

            print('Loss D: {}'.format(loss_D_GAN.item()),end=' ')
            optim_D.step()

            # Train G
            # GAN loss
            optim_G.zero_grad()

            pred_fake = disc(adv_images)
            loss_G_fake = F.mse_loss(pred_fake, real_labels)
            loss_G_fake.backward(retain_graph=True)
            print('Loss G fake {}'.format(loss_G_fake.item()),end=' ')

            loss_perturbed = hinge_loss(perturbation)
            print('Loss perturbed: {}'.format(loss_perturbed.item()),end=' ')
            loss_adv = adv_target_loss(cls_net,tgt_class,adv_images)
            print('Loss adv: {}'.format(loss_adv.item()))
            alpha = 3.;beta =1.
            loss_G = alpha*loss_adv+beta*loss_perturbed
            loss_G.backward()
            optim_G.step()
    
    torch.save(gen.state_dict(),'gen_target.pth')


if __name__=='__main__':
    gen = Generator(1,1)
    disc = Discriminator(1)
    cls_net = MNIST_Classifier()
    cls_net.load_state_dict(torch.load('mnist_classifier.pth'))
    cls_net.eval()
    dataset = MNIST('./data',train=True,transform=transforms.ToTensor())
    dataloader = DataLoader(dataset,batch_size=128,shuffle=True)
    adversarial_training_targeted(dataloader,gen,disc,cls_net,0,epochs=60)