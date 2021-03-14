import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

class MNIST_Classifier(nn.Module):
    def __init__(self):
        super(MNIST_Classifier,self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,padding=1),
            nn.ReLU()
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32,32,kernel_size=3,padding=1),
            nn.ReLU()
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.ReLU()
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU()
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.linear1 = nn.Sequential(
            nn.Linear(7*7*64,200),
            nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(200,200),
            nn.ReLU()
        )
        self.linear3 = nn.Linear(200,10)
        self.drop = nn.Dropout(0.5)

    def forward(self,inputs):
        
        x = self.conv_block1(inputs)
        x = self.conv_block2(x)
        x = self.max_pool(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.max_pool(x)
        x = x.view(-1,7*7*64)
        x = self.linear1(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)
        outputs = self.linear3(x)
        return outputs

def train():
    epochs = 40
    cls_net = MNIST_Classifier()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cls_net.to(device)
    cls_net.train()
    mnist_train = MNIST('./data',train=True,transform=transforms.ToTensor())
    train_dataloader = DataLoader(mnist_train,batch_size=256,shuffle=True)
    optimizer = torch.optim.Adam(cls_net.parameters(),lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for i,data in enumerate(train_dataloader,0):
            imgs,targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = cls_net(imgs)
            losses = criterion(outputs,targets)
            print('Step {}: Loss {}'.format(i,losses.item()))
            losses.backward()
            optimizer.step()
    torch.save(cls_net.state_dict(),'mnist_classifier.pth')

def eval():
    cls_net = MNIST_Classifier()
    cls_net.load_state_dict(torch.load('mnist_classifier.pth'))
    cls_net.eval()
    mnist_test = MNIST('./data',train=False,transform=transforms.ToTensor())
    test_dataloader = DataLoader(mnist_test,batch_size=256,shuffle=True)
    count = 0
    for imgs,targets in test_dataloader:
        outputs = cls_net(imgs)
        preds = torch.argmax(outputs,dim=-1)
        count += torch.sum(preds==targets)
    print(count)

if __name__=='__main__':
    eval()