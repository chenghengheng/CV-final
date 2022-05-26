import argparse

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from vit_pytorch import ViT

from load import load
from resnet import ResNet
from util import optimize, evaluate, save_status

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default = 128, type = int)
    parser.add_argument("--num_epoch", default = 60, type = int)
    parser.add_argument("--model", default = 'transformer', type = str)
    parser.add_argument("--lr", default = 0.02, type = float)
    parser.add_argument("--milestones", default = range(20, 60, 20), type = list)
    parser.add_argument("--gamma", default = 0.2, type = float)
    parser.add_argument("--momentum", default = 0.9, type = float)
    parser.add_argument("--lambd", default = 5e-4, type = float)
    parser.add_argument("--mode", default = 'baseline', type = str)
    args = parser.parse_args()

    train_loader = load(train = True, batch_size = args.batch_size, shuffle = True)
    test_loader = load(train = False, batch_size = args.batch_size, shuffle = False)

    print('number of iterations:', args.num_epoch * len(train_loader))
    print('number of iterations per epoch:', len(train_loader))
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == 'resnet':
        model = ResNet(num_classes = 100).to(device)
    else:
        model = ViT(image_size = 32, patch_size = 16, num_classes = 100,
                    dim = 512, depth = 2, heads = 8, mlp_dim = 1024, dim_head = 64).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.lambd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = args.milestones, gamma = args.gamma)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(args.model)

    print('Epoch\tTrain top1\tTrain top5\tTest top1\tTest top5\t')
    for ind_epoch in range(args.num_epoch):
        optimize(model, criterion, optimizer, train_loader, args.mode, device)
        scheduler.step()

        train_acc_t1, train_acc_t5, train_loss = evaluate(model, criterion, train_loader, device)
        test_acc_t1, test_acc_t5, test_loss = evaluate(model, criterion, test_loader, device)

        writer.add_scalars('top1', {'train': train_acc_t1, 'test': test_acc_t1}, ind_epoch)
        writer.add_scalars('top5', {'train': train_acc_t5, 'test': test_acc_t5}, ind_epoch)
        writer.add_scalars('loss', {'train': train_loss, 'test': test_loss}, ind_epoch)

        print('%2d\t%.5f\t\t%.5f\t\t%.5f\t\t%.5f' %
              (ind_epoch + 1, train_acc_t1, train_acc_t5, test_acc_t1, test_acc_t5))

    save_status(model, optimizer, args.model + '.pth')

    writer.flush()
    writer.close()
