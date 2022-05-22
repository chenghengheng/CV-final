import numpy as np
import torch
import matplotlib.pyplot as plt

from load import load


def cutout(inputs, length = 16, device = torch.device('cpu')):
    n, c, h, w = inputs.shape

    for i in range(n):
        mask = torch.ones(h, w).to(device)

        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.

        mask = mask.expand(c, h, w)
        inputs[i] = inputs[i] * mask

    return


if __name__ == '__main__':
    train_loader = load()

    for inputs, targets in train_loader:
        cutout(inputs)

        print(inputs[0])
        print(targets[0])

        print(inputs[0].shape)
        print(inputs[0].max())
        print(inputs[0].min())

        for i in range(3):
            img = np.transpose(inputs[i], (1, 2, 0))
            plt.imshow(img * 0.5 + 0.5)
            plt.axis('off')
            plt.savefig('cutout' + str(i) + '.png')
            plt.show()

        break