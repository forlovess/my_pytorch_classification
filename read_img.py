import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T


class My_Data(data.Dataset):

    def __init__(self, root, transforms=None, train=True, test=False):
        '''
        目标：获取所有图片路径，并根据训练、验证、测试划分数据
        '''
        self.test = test
        classs = os.listdir(root)
        imgs = []
        labels = []
        for idx, folder in enumerate(classs):
            cate = os.path.join(root, folder)
            for img_num, im in enumerate(os.listdir(cate)):
                img_path = os.path.join(cate, im)
                #打包图片路径（转换为list）
                imgs.append(img_path)
                #打包标签路径（转换为list）
                labels.append(idx)
        if self.test:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:

            imgs = list(zip(imgs , labels))
            #将图片路径与标签打包成一个list

        imgs_num = len(imgs)

        # shuffle imgs
        np.random.seed(100)
        imgs = np.random.permutation(imgs)

        # 划分训练、验证集，验证:训练 = 3:7
        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7 * imgs_num)]
        else:
            self.imgs = imgs[int(0.7 * imgs_num):]

        if transforms is None:

            # 数据转换操作，测试验证和训练的数据转换有所区别
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

            # 测试集和验证集不用数据增强
            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize(32),
                    T.CenterCrop(32),
                    T.ToTensor(),
                    normalize
                ])
                # 训练集需要数据增强
            else:
                self.transforms = T.Compose([
                    T.Resize(32),
                    T.RandomResizedCrop(32),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self,index):
        '''
        返回一张图片的数据
        对于测试集，没有label，返回图片id，如1000.jpg返回1000
        送入一个batch_size的数据
        '''

        img_lables = self.imgs[index]
        img_path = img_lables[0]

        if self.test:
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            label = int(img_lables[1])

        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        '''
        返回数据集中所有图片的个数
        '''
        return len(self.imgs)

