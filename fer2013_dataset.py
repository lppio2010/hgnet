import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision

class Fer2013Dataset(Dataset):
    def __init__(self, transforms):
        dataset = np.load('../img_db/data_48.npz')
        self.imgs = dataset['x_train']
        self.labels = dataset['y_train']
        self.num_class = 7
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def preprocess(self, x):
        mean = torch.mean(x)
        std = torch.std(x)
        x = (x - mean) / std
        return x

    def __getitem__(self, idx):
        img = self.imgs[idx]
        tensor_img = self.transforms(img)
        tensor_img = self.preprocess(tensor_img)
        tensor_label = torch.from_numpy(self.labels[idx]).type(torch.int64).item()
        sample = {'image': tensor_img, 'label': tensor_label}
        return sample

class Fer2013ValidDataset(Dataset):
    def __init__(self, transforms):
        dataset = np.load('../img_db/data_48_test.npz')
        self.imgs = dataset['x_test']
        self.labels = dataset['y_test']
        self.num_class = 7
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def preprocess(self, x):
        mean = torch.mean(x)
        std = torch.std(x)
        x = (x - mean) / std
        return x

    def __getitem__(self, idx):
        img = self.imgs[idx]
        tensor_img = self.transforms(img)
        tensor_img = self.preprocess(tensor_img)
        tensor_label = torch.from_numpy(self.labels[idx]).type(torch.int64).item()
        sample = {'image': tensor_img, 'label': tensor_label}
        return sample