import torch
from torch.nn import functional as F
import torch.nn as nn

OMEGA = 0.56714329040978387299997  # W(1, 0)
EXPN1 = 0.36787944117144232159553  # exp(-1)

def conv(X,C):
    kernel = C.shape[2]
    padding = kernel//2
    X = X.unsqueeze(1)
    C = C.unsqueeze(1)
    convolve = [F.conv2d(X[i],C[i],padding=padding) for i in range(X.shape[0])]
    convolve = torch.cat(convolve,dim=0)
    return convolve

def lambertw(z0, tol=1e-5): 
    # this is a direct port of the scipy version for the 
    # k=0 branch for *positive* z0 (z0 >= 0)

    # skip handling of nans
    if torch.isnan(z0).any(): 
        raise NotImplementedError

    w0 = z0.new(*z0.size())
    # under the assumption that z0 >= 0, then I_branchpt 
    # is never used. 
    I_branchpt = torch.abs(z0 + EXPN1) < 0.3
    I_pade0 = (-1.0 < z0)*(z0 < 1.5)
    I_asy = ~(I_branchpt | I_pade0)
    if I_pade0.any(): 
        z = z0[I_pade0]
        num = torch.Tensor([
            12.85106382978723404255,
            12.34042553191489361902,
            1.0
        ]).to(z.device)
        denom = torch.Tensor([
            32.53191489361702127660,
            14.34042553191489361702,
            1.0
        ]).to(z.device)
        w0[I_pade0] = z*evalpoly(num,2,z)/evalpoly(denom,2,z)

    if I_asy.any(): 
        z = z0[I_asy]
        w = torch.log(z)
        w0[I_asy] = w - torch.log(w)

    # split on positive and negative, 
    # and ignore the divergent series case (z=1)
    w0[z0 == 1] = OMEGA
    I_pos = (w0 >= 0)*(z0 != 1)
    I_neg = (w0 < 0)*(z0 != 1)
    if I_pos.any(): 
        w = w0[I_pos]
        z = z0[I_pos]
        for i in range(100): 
            # positive case
            ew = torch.exp(-w)
            wewz = w - z*ew
            wn = w - wewz/(w + 1 - (w + 2)*wewz/(2*w + 2))

            if (torch.abs(wn - w) < tol*torch.abs(wn)).all():
                break
            else:
                w = wn
        w0[I_pos] = w

    if I_neg.any(): 
        w = w0[I_neg]
        z = z0[I_neg]
        for i in range(100):
            ew = torch.exp(w)
            wew = w*ew
            wewz = wew - z
            wn = w - wewz/(wew + ew - (w + 2)*wewz/(2*w + 2))
            if (torch.abs(wn - w) < tol*torch.abs(wn)).all():
                break
            else:
                w = wn
        w0[I_neg] = wn
    return w0

def evalpoly(coeff, degree, z): 
    powers = torch.arange(degree,-1,-1).float().to(z.device)
    return ((z.unsqueeze(-1)**powers)*coeff).sum(-1)

def lamw(x): 
    I = x > 1e-10
    y = torch.clone(x)
    y[I] = lambertw(x[I])
    return y

def collapse3(x):
    # Return 2 dimension matrix (only keep the batch dimension, collapse all other dimension)
    return x.view(x.shape[0],-1)

def dotmatrix(x,y):
    # input x and y are tensors (greater than 2 dimensions)
    mul = torch.matmul(collapse3(x),collapse3(y).transpose(0,1))
    return torch.diagonal(mul,0)

def lagrarian(alpha,beta,exp_alpha,exp_beta,X,Y,psi,K,lambd,epsilon):
    """Compute the lagrarian
    The lagrarian can be computed as:
    g = -0.5*|beta|^2/lambd-psi*epsilon+alpha^TX+beta^TY-exp(alpha)exp(-psiC-1)exp(beta)

    Args:
        alpha ([type]): [description]
        beta ([type]): [description]
        exp_alpha ([type]): [description]
        X:
        Y:
        exp_beta ([type]): [description]
        psi ([type]): [description]
        K ([type]): [description]
    """
    return -0.5*dotmatrix(beta,beta)/lambd-psi*epsilon-dotmatrix(torch.clamp(alpha,max=1e10),X)-dotmatrix(torch.clamp(beta,max=1e10),Y)\
           -dotmatrix(exp_alpha,conv(exp_beta,K))

def wasserstein_kernel(p=2,kernel_size=7):
    """The wasserstein distance

    Args:
        p (int, optional): [description]. Defaults to 2.
        kernel_size (int, optional): [description]. Defaults to 7.
    """
    if kernel_size%2 != 1:
        raise ValueError("Kernel size needs to be odd")
    center = kernel_size//2
    C = [[((i-center)**2+(j-center)**2)**(p/2) for j in range(kernel_size)] for i in range(kernel_size)]
    C = torch.tensor(C)
    return C.unsqueeze(0).unsqueeze(0)

def project_sinkhorn(X,Y,C,epsilon,lambd,max_iter=50):
    """Implement projected sinkhorn algorithm given in
    the paper: Wasserstein Adversarial Examples via Projected Sinkhorn Iterations.


    Args:
        X : original image
        Y : The image we want to project it into wasserstein ball centered by the original image.
        C : Cost matrix
        epsilon : radius of Wasserstein ball
        lambd : trade-off coefficient in Projected sinkhorn problem
    """
    
    # Initialize alpha,beta,psi
    size = X.size()
    m = X.shape[1]*X.shape[2]*X.shape[3]
    alpha = torch.log(torch.ones(size)/m)
    beta = torch.log(torch.ones(size)/m)
    u = torch.exp(-alpha)
    v = torch.exp(-beta)
    psi = torch.ones(X.shape[0])

    K = torch.exp(-psi.view(-1,1,1,1)*C-1)
    old_lagrange = -float('inf')
    i = 0 # number of iteration
    while True:
        alpha = -torch.log(X)+torch.log(conv(v,K))
        u = torch.exp(-alpha)
        beta = -lambd*Y + lamw(lambd*torch.exp(lambd*Y)*conv(u,K))
        v = torch.exp(-beta)

        # Newton step
        g = -epsilon+dotmatrix(u,conv(v,K*C))
        h = -dotmatrix(u,conv(v,C*C*K))
        delta = g/h
        t = torch.ones_like(delta)
        neg = (psi-t*delta)<0
        while neg.any() and t.min().item()>1e-2:
            t[neg] /=2
            neg = psi-t*delta < 0
        psi = torch.clamp(psi-t*delta,min=0)
        K = torch.exp(-psi.view(psi.shape[0],1,1,1)*C-1)
        lagrange = lagrarian(alpha,beta,u,v,X,Y,psi,K,lambd,epsilon)
        i+=1
        if i>max_iter or torch.sum(torch.abs(lagrange-old_lagrange)<=1e-4):
            break
        old_lagrange = lagrange
    return Y+beta/lambd