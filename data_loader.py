import torch.utils.data as data
from PIL import Image

class mnist_loader(data.Dataset):
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
