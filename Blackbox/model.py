import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms

def jacobian_datagen(model,imgs,labels,lambd=0.1):
    '''
    Generate new synthetic data by Jacobian Data Augmentation
    Params:
    model: substitute model
    imgs: batch of tensor
    labels: corresponding label
    Recall the update of the Jacobian data generation
    S_{p+1} = x+lambd_{p+1}sign(JF[O(x)])
    '''
    crarfted_samples = []
    for i in range(len(imgs)):
        img = imgs[i].unsqueeze(0)
        target = labels[i]
        img.requires_grad = True
        output = model(img)

        selected_output = output[0,target]
        selected_output.backward()
        new_sample = img+lambd*torch.sign(img.grad)
        new_sample = torch.clamp(new_sample.detach(),0,1)
        crarfted_samples.append(new_sample)
    
    # Combine
    crarfted_samples = torch.cat(crarfted_samples,dim=0)
    return torch.cat([imgs,crarfted_samples],dim=-1)

class LRModel(nn.Module):
    def __init__(self):
        super(LRModel,self).__init__()
        self.linear_layer = nn.Linear(784,10)
    
    def forward(self,inputs):
        imgs = inputs.view(inputs.shape[0],-1)
        outputs = self.linear_layer(imgs)
        return outputs

@torch.no_grad()
def predict(imgs,model):
    outputs = model(imgs)
    return torch.argmax(outputs,dim=-1)

def creating_labels(imgs,oracle,batch_size):
    '''
    Creating labels for the datasets
    Params: 
    imgs: tensor of images
    oracle: black-box model
    batch_size
    '''
    labels = []
    steps = int(np.ceil(len(imgs)/batch_size))
    for step in steps:
        if step+1<len(imgs):
            img = imgs[step*batch_size:(step+1)*batch_size]
            labels.append(predict(imgs,oracle))
        else:
            img = imgs[step*batch_size:]
        labels = torch.cat(labels)
        return labels

def train_nets(model,imgs,labels,batch_size,lr=1e-2,sub_epochs=6):
    
    '''
    Training the model
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    criterion = nn.CrossEntropyLoss()

    steps = int(np.ceil(len(imgs)/batch_size))
    for epoch in range(sub_epochs):
        print('Epoch {}:'.format(epoch))
        indices = torch.randperm(len(imgs))
        for step in range(steps):
            index = indices[step*batch_size:(step+1)*batch_size]
        else:
            index = indices[step*batch_size]
        
        images = imgs[index].to(device)
        targets = labels[index].to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs,targets)
        print('Loss at step {}: {}'.format(step,loss.item()))
        loss.backward()
        optimizer.step()
    
    return model

def one_subs_epoch(imgs,oracle,batch_size=16,lr=1e-2,sub_epochs=10,lambd=0.1):
    '''
    This is what's happening at one substitute epoch
    '''
    model = LRModel()
    labels = creating_labels(imgs,oracle,batch_size)
    train_nets(model,imgs,labels,batch_size,lr=lr,sub_epochs=sub_epochs)
    
    new_samples = jacobian_datagen(model,imgs,labels,lambd=lambd)
    return model,new_samples,labels


if __name__=='__main__':
    from cls_model import MNIST_Classifier
    import random
    oracle = MNIST_Classifier()
    oracle.load_state_dict(torch.load('../NAA/saved_models/mnist_classifier.pth'))
    oracle.to('cuda')
    oracle.eval()
    dataset = MNIST('../data',train=False,transform=transforms.ToTensor())
    cls_idx = [[] for i in range(10)]
    for i in range(10000):
        _,target = dataset.__getitem__(i)
        cls_idx[target].append(i)
    for i in range(10):
        random.shuffle(cls_idx[i])
    list_samples = []
    for i in range(10):
        for j in range(10):
            img,_ = dataset.__getitem__(cls_idx[i][j])
            list_samples.append(img)
    
    list_samples = torch.cat(list_samples)
    samples = list_samples
    for epoch in range(7):
        print('Substitute epoch {}'.format(epoch))
        model,samples,labels = one_subs_epoch(samples,oracle)