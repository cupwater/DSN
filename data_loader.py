import torch.utils.data as data
from PIL import Image
import os


class GetLoader(data.Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform

        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data



class mnist_m_loader(data.Dataset):
    def __init__(self, data, label, train=True, transform=None):
        self.transform = transform
        self.n_data = len(label)
        self.train = train  # training set or test set

        self.img_labels = label
        self.img_data   = data

    def __getitem__(self, item):

        imgs   = self.img_data[item]
        imgs = imgs.numpy()
        imgs = Image.fromarray(imgs, 'RGB')

        labels = self.img_labels[item]

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data
