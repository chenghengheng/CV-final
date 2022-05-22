import numpy as np
import torch
import matplotlib.pyplot as plt

from load import load


def cutmix(inputs, targets, alpha = 1):
    n, c, h, w = inputs.shape

    index = torch.randperm(n)
    lambd = np.random.beta(alpha, alpha)

    r = np.sqrt(1 - lambd)

    y = np.random.randint(h)
    x = np.random.randint(w)

    y1 = np.clip(y - int(h * r) // 2, 0, h)
    y2 = np.clip(y + int(h * r) // 2, 0, h)
    x1 = np.clip(x - int(w * r) // 2, 0, w)
    x2 = np.clip(x + int(w * r) // 2, 0, w)

    lambd = 1 - (x2 - x1) * (y2 - y1) / (h * w)

    inputs_cutmix = inputs.clone()
    inputs_cutmix[:, :, x1: x2, y1: y2] = inputs_cutmix[index, :,  x1: x2,  y1: y2]
    targets_cutmix = targets[index]

    return inputs_cutmix, targets_cutmix, lambd, index


if __name__ == '__main__':
    train_loader = load()

    for inputs, targets in train_loader:
        inputs_cutmix, targets_cutmix, lambd, index = cutmix(inputs, targets)

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
            plt.show()

            img = np.transpose(inputs_cutmix[i], (1, 2, 0))
            plt.imshow(img * 0.5 + 0.5)
            plt.axis('off')
            plt.savefig('cutmix' + str(i) + '.png')
            plt.show()

        break