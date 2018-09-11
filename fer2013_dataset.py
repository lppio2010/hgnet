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

    def one_hot(self, label):
        label_vec = torch.zeros(self.num_class, dtype=torch.int64)
        label_vec[label] = 1
        return label_vec

    def preprocess(self, x):
        mean = np.mean(x)
        x = x - mean
        return x

    def __getitem__(self, idx):
        img = self.preprocess(self.imgs[idx])
        tensor_img = torch.from_numpy(np.reshape(img, (1, 48, 48))).type(torch.float)
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

    def one_hot(self, label):
        label_vec = torch.zeros(self.num_class, dtype=torch.int64)
        label_vec[label] = 1
        return label_vec

    def preprocess(self, x):
        mean = np.mean(x)
        x = x - mean
        return x

    def __getitem__(self, idx):
        img = self.preprocess(self.imgs[idx])
        tensor_img = torch.from_numpy(np.reshape(img, (1, 48, 48))).type(torch.float)
        tensor_label = torch.from_numpy(self.labels[idx]).type(torch.int64).item()
        sample = {'image': tensor_img, 'label': tensor_label}
        return sample