import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader


def load(root = '../data/', train = True, batch_size = 128, shuffle = True, num_workers = 4):
    torch.manual_seed(0)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomCrop(32, padding = 4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])])

    dataset = datasets.CIFAR100(root = root, train = train, download = True, transform = transform)
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
    return loader


if __name__ == '__main__':
    train_loader = load()

    for inputs, targets in train_loader:
        print(inputs[0])
        print(targets[0])

        print(inputs[0].shape)
        print(inputs[0].max())
        print(inputs[0].min())

        for i in range(3):
            img = np.transpose(inputs[i], (1, 2, 0))
            plt.imshow(img * 0.5 + 0.5)
            plt.axis('off')
            plt.savefig('baseline' + str(i) + '.png')
            plt.show()

        break
