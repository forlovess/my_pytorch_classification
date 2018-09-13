# coding:utf8
from torchvision import datasets, models
from read_img import My_Data
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

train_data_root = 'D:\dataset\\flowers'
batch_size = 16
load_model_path = './model/'

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 5)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)
from torch import optim
criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
def train():

    train_data = My_Data(train_data_root, train=True)
    val_data = My_Data(train_data_root, train=False)
    train_dataloader = DataLoader(train_data, batch_size,
                                  shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_data,batch_size,
                                shuffle=False, num_workers=0)
    torch.set_num_threads(8)
    for epoch in range(20):

        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):

            # 输入数据
            inputs, labels = data
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            net.to(device)
            images = inputs.to(device)
            labels = labels.to(device)
            # output = net(images)
            # loss = criterion(output, labels)
            # 梯度清零
            optimizer.zero_grad()

            # forward + backward
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # 更新参数
            optimizer.step()

            # 打印log信息
            # loss 是一个scalar,需要使用loss.item()来获取数值，不能使用loss[0]
            running_loss += loss.item()
            print('[%d, %5d] loss: %.3f'% (epoch + 1, i + 1, running_loss ))
            running_loss = 0.0
    print('Finished Training')
train()