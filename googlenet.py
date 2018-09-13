import torch as t
from torch.nn import  functional as F

class Inception_Block(t.nn.Module):
    def __init__(self,input_channal,output_channal,stride=1,auxiliary=False):
        super(Inception_Block,self).__init__()
        self.path_A = t.nn.Sequential(
            t.nn.Conv2d(input_channal,output_channal,1,stride,1,bias=False),
            t.nn.BatchNorm2d(output_channal),
            t.nn.ReLU(output_channal))
        self.path_B = t.nn.Sequential(
            t.nn.Conv2d(input_channal, output_channal, 1, stride, 1, bias=False),
            t.nn.BatchNorm2d(output_channal),
            t.nn.ReLU(output_channal),
            t.nn.Conv2d(output_channal, output_channal, 3, 1, 1, bias=False),
            t.nn.BatchNorm2d(output_channal),
            t.nn.ReLU(output_channal)
        )
        self.path_C = t.nn.Sequential(
            t.nn.Conv2d(input_channal, output_channal, 1, stride, 1, bias=False),
            t.nn.BatchNorm2d(output_channal),
            t.nn.ReLU(output_channal),
            t.nn.Conv2d(output_channal, output_channal, 5, 1, 1, bias=False),
            t.nn.BatchNorm2d(output_channal),
            t.nn.ReLU(output_channal),
        )
        self.path_D = t.nn.Sequential(
            t.nn.MaxPool2d(3, 1, 1),
            t.nn.Conv2d(input_channal, output_channal, 1, stride, 1, bias=False),
            t.nn.BatchNorm2d(output_channal),
            t.nn.ReLU(output_channal))
        self.auxiliary = auxiliary

        if auxiliary:
            self.auxiliary_layer = t.nn.Sequential(
                t.nn.AvgPool2d(5, 3),
                t.nn.Conv2d(input_channal, 128, 1),
                t.nn.ReLU())

    def forward(self, x,train=False):
        convA_out = self.path_A(x)
        convB_out = self.path_B(x)
        convC_out = self.path_C(x)
        pool_conv_out = self.path_D(x)
        outputs = t.cat([convA_out, convB_out, convC_out, pool_conv_out], 1)
        if self.auxiliary:
            if train:
                outputs2 = self.auxiliary_layer(x)
            else:
                outputs2 = None
            return outputs, outputs2
        else:
            return outputs


class GoogleNet(t.nn.Module):
    def __init__(self,num_classes = 1000,n = 1):
        super(GoogleNet,self).__init__()
        self.stem_layer = t.nn.Sequential(
            t.nn.Conv2d(3, 64, 7, 2, 3),
            t.nn.ReLU(),
            t.nn.MaxPool2d(3, 2, 1),
            t.nn.Conv2d(64, 64, 1),
            t.nn.ReLU(),
            t.nn.Conv2d(64, 192, 3, 1, 1),
            t.nn.ReLU(),
            t.nn.MaxPool2d(3, 2, 1)
        )
        self.layer1  = self.Inception_Block( 64, 64, 3*n)
        self.layer2 = self.Inception_Block(64, 128, 4*n, stride=2)
        self.layer3 = self.Inception_Block(128, 256, 6*n, stride=2)
        self.layer4 = self.Inception_Block(256, 512, 3*n, stride=2)
        self.fc = t.nn.Linear(512, num_classes)

    def forward(self, inputs, train=False):
        outputs = self.stem_layer(inputs)
        outputs = self.inception_layer1(outputs)
        outputs = self.inception_layer2(outputs)
        # outputs,outputs2 = self.inception_layer3(outputs)
        # if train:
        # B,128,4,4 => B,128*4*4
        #    outputs2 = self.auxiliary_layer(outputs2.view(inputs.size(0),-1))
        outputs = self.inception_layer3(outputs)
        outputs = self.dropout(outputs)
        outputs = outputs.view(outputs.size(0), -1)  # 동일 : outputs = outputs.view(batch_size,-1)
        outputs = self.output_layer(outputs)

        # if train:
        #   return outputs, outputs2
        return outputs
