import torch as t
from torch.nn import  functional as F

class ResidualBlock(t.nn.Module):
    def __init__(self,input_channal,output_channal,stride=1,shrotcut = None):
        super(ResidualBlock,self).__init__()
        self.left = t.nn.Sequential(
            t.nn.Conv2d(input_channal,output_channal,3,stride,1,bias=False),
            t.nn.BatchNorm2d(output_channal),
            t.nn.ReLU(output_channal),
            t.nn.Conv2d(output_channal,output_channal,3,1,1,bias=False)
        )
        self.right = shrotcut
    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)
class Resnet(t.nn.Module):
    def __init__(self,num_classes = 1000,n = 1):
        super(Resnet,self).__init__()
        self.pre_layer = t.nn.Sequential(
            t.nn.Conv2d(3,64,(7,7),2,3,bias=False),
            t.nn.BatchNorm2d(64),
            t.nn.ReLU(inplace=True),
            t.nn.MaxPool2d(3,2,1))
        self.layer1  = self._make_layer( 64, 64, 3*n)
        self.layer2 = self._make_layer(64, 128, 4*n, stride=2)
        self.layer3 = self._make_layer(128, 256, 6*n, stride=2)
        self.layer4 = self._make_layer(256, 512, 3*n, stride=2)
        self.fc = t.nn.Linear(512, num_classes)

    def _make_layer(self,input_channal,output_channal,block_num, stride=1):
        shortcut = t.nn.Sequential(
            t.nn.Conv2d(input_channal,output_channal,1,stride,bias=False),
            t.nn.BatchNorm2d(output_channal))
        layers = []
        layers.append(ResidualBlock(input_channal,output_channal, stride, shortcut))
        for i in range(1, block_num):
            layers.append(ResidualBlock(output_channal, output_channal))
        return t.nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre_layer(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)
model = Resnet()
print(model)
input  = t.autograd.Variable(t.randn(1, 3, 224, 224))
o = model(input)
print(o.size())