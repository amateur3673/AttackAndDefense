import torch
import torch.nn as nn

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