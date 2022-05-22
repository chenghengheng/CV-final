import torch

from cutout import cutout
from mixup import mixup
from cutmix import cutmix


def get_num_parameters(model):
    num_parameters = 0
    for parameter in model.parameters():
        num_parameters += torch.prod(torch.tensor(parameter.shape))
    return num_parameters


def optimize(model, criterion, optimizer, train_loader, mode, device):
    model.train()

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if mode == 'baseline':
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        elif mode == 'cutout':
            cutout(inputs, device = device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        elif mode == 'mixup':
            inputs_mixup, targets_mixup, lambd, index = mixup(inputs, targets)
            outputs = model(inputs_mixup)
            loss = lambd * criterion(outputs, targets) + (1 - lambd) * criterion(outputs, targets_mixup)
        else:  # mode == 'cutmix'
            inputs_cutmix, targets_cutmix, lambd, index = cutmix(inputs, targets)
            outputs = model(inputs_cutmix)
            loss = lambd * criterion(outputs, targets) + (1 - lambd) * criterion(outputs, targets_cutmix)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(model, criterion, data_loader, device):
    model.eval()

    count = 0
    losses = 0
    correct_t1 = 0
    correct_t5 = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        count += inputs.shape[0]
        _, pred_t1 = outputs.max(dim = 1)
        _, pred_t5 = torch.topk(outputs, k = 5, dim = 1)
        correct_t1 += pred_t1.eq(targets).sum().item()
        correct_t5 += pred_t5.eq(torch.unsqueeze(targets, 1).repeat(1, 5)).sum().item()
        losses += loss.item() * inputs.shape[0]

    return correct_t1 / count, correct_t5 / count, losses / count


def save_status(model, optimizer, path):
    save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(save_dict, path)


def load_status(model, optimizer, path):
    load_dict = torch.load(path)
    model.load_state_dict(load_dict['model'])
    optimizer.load_state_dict(load_dict['optimizer'])