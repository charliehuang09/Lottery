import torch
import torch.nn as nn

class Model(torch.nn.Module):

    def __init__(self, ch=32):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(1, ch, (3, 3))
        self.conv2 = nn.Conv2d(ch, ch, (3, 3))
        self.conv3 = nn.Conv2d(ch, ch, (3, 3))
        self.conv4 = nn.Conv2d(ch, ch, (3, 3))
        self.conv5 = nn.Conv2d(ch, ch, (3, 3))
        self.maxpool = nn.MaxPool2d(3, 3)
        self.flatten = nn.Flatten()
        self.end = nn.Linear(1152, 10)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.20)


    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.activation(x)

        x = self.conv3(x)
        x = self.activation(x)

        x = self.conv4(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv5(x)
        x = self.activation(x)
        x = self.maxpool(x)
        
        x = self.flatten(x)
        x = self.end(x)
        x = self.sigmoid(x)

        return x
