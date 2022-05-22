import numpy as np
import torch
import matplotlib.pyplot as plt

from load import load


def mixup(inputs, targets, alpha = 1):
    index = torch.randperm(inputs.shape[0])
    lambd = np.random.beta(alpha, alpha)

    inputs_mixup = lambd * inputs + (1 - lambd) * inputs[index]
    targets_mixup = targets[index]

    return inputs_mixup, targets_mixup, lambd, index


if __name__ == '__main__':
    train_loader = load()

    for inputs, targets in train_loader:
        inputs_mixup, targets_mixup, lambd, index = mixup(inputs, targets)

        print(inputs[0])
        print(targets[0])

        print(inputs[0].shape)
        print(inputs[0].max())
        print(inputs[0].min())

        for i in range(3):
            img = np.transpose(inputs[i], (1, 2, 0))
            plt.imshow(img * 0.5 + 0.5)
            plt.axis('off')
            plt.show()

            img = np.transpose(inputs[index[i]], (1, 2, 0))
            plt.imshow(img * 0.5 + 0.5)
            plt.axis('off')
            plt.savefig('baseline' + str(i + 3) + '.png')
            plt.show()

            img = np.transpose(inputs_mixup[i], (1, 2, 0))
            plt.imshow(img * 0.5 + 0.5)
            plt.axis('off')
            plt.savefig('mixup' + str(i) + '.png')
            plt.show()

        break