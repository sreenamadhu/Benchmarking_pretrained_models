import numpy as np
import os
import sys
from PIL import Image
import torch
import torch.utils.data as data

class Dataloader(data.Dataset):

    def __init__(self, data_dir = '/home/sreena/Documents/benchmarks/data', filename = 'test.txt',transform= None, flag = 'part1'):

        self.data_names = []
        self.labels = []
        self.flag = flag
        self.transform = transform
        self.imgs = []

        with open(os.path.join(data_dir,filename)) as f:

            for line in f:
                line = line.strip('\r\n').split(' ')

                if self.flag == 'part1':
                    line1 = line[0].split('/')

                else:
                    line1 = line[1].split('/')

                self.data_names.append(os.path.join(data_dir,line1[-2],line1[-1]))
                self.labels.append(int(line[2]))
                self.imgs.append((os.path.join(data_dir,line1[-2],line1[-1]), int(line[2])))

    def __getitem__(self,index):


        img_path = self.data_names[index]
        img = Image.open(img_path).convert('RGB')
        """
        Rotate the image for panel wise A8 trained models
        """
        # width,height = img.size

        # if (width-20 > height):
        #     imgNew = img.rotate(90,expand = True)
        #     img = imgNew.resize((height,width),Image.ANTIALIAS)


        if self.transform is not None:
            img = self.transform(img)

        target = self.labels[index]

        return img, target, index

    def __len__(self):

        return len(self.imgs)

class New_Dataloader(data.Dataset):

    def __init__(self,data_dir = '/home/sreena/Desktop/Ran/dataset/', filename = 'pair_test_crop.txt', transform = None, flag = 'part1', pano = 'A_8'):

        self.data_names = []
        self.labels = []
        self.flag = flag
        self.transform = transform
        self.imgs = []
        with open(os.path.join(data_dir,filename)) as f:

            for line in f:
                line = line.strip('\r\n').split(',')
                if self.flag == 'part1':
                    line1 = line[0]
                else:
                    line1 = line[0]

                if pano:
                    if pano == line1[0:3]:
                        self.data_names.append(os.path.join(data_dir,line1))
                        self.labels.append(int(line[2]))
                        self.imgs.append((os.path.join(data_dir,line1)))
                else:
                    self.data_names.append(os.path.join(data_dir,line1))
                    self.labels.append(int(line[2]))
                    self.imgs.append((os.path.join(data_dir,line1)))



    def __getitem__(self,index):

        img_path = self.data_names[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        target = self.labels[index]
        return img,target,index

    def __len__(self):
        return len(self.imgs)




class Feature_loader(data.Dataset):

    def __init__(self, data_dir = '/home/sreena/Documents/benchmarks/new_data/', flag = 'train'):
        
        self.labels = []
        self.features = []

        feature1 = torch.load(data_dir + 'features/' + flag + '1_feature.pt')
        feature2 = torch.load(data_dir + 'features/' + flag + '2_feature.pt')

        features = torch.cat((feature1,feature2),1)


        with open(data_dir + flag + '1.txt') as f:

            for line in f:
                line = line.strip('\r\n').split(' ')
                labels.append(int(line[1]))

        labels = np.array(labels)
        range_labels = np.array(range(len(labels)))
        final_labels = np.zeros((len(labels),2))
        final_labels[range_labels,labels] = 1
        self.labels = final_labels


    def __getitem__(self,index):


        img = self.features[index]

        target = self.labels[index]

        return img, target


    def __len__(self):

        return len(self.features)



