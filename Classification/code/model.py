import torch
from torch import nn

from util import get_num_parameters


def conv1x1(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, padding = 0, bias = False)


def conv3x3(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, stride = 1):
        super().__init__()
        out_channels = in_channels * stride

        self.stride = stride
        self.relu = nn.ReLU(inplace = True)
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if self.stride > 1:
            self.conv3 = conv1x1(in_channels, out_channels, stride)
            self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, inputs):
        identity = inputs

        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.relu(outputs)

        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)

        if self.stride > 1:
            identity = self.conv3(identity)
            identity = self.bn3(identity)

        outputs += identity
        outputs = self.relu(outputs)

        return outputs


class ResNet(nn.Module):
    def __init__(self, num_classes = 100):
        super().__init__()

        self.conv = conv3x3(in_channels = 3, out_channels = 64)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        # self.pool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.block1 = BasicBlock(in_channels = 64, stride = 1)
        self.block2 = BasicBlock(in_channels = 64, stride = 1)
        self.block3 = BasicBlock(in_channels = 64, stride = 2)
        self.block4 = BasicBlock(in_channels = 128, stride = 1)
        self.block5 = BasicBlock(in_channels = 128, stride = 2)
        self.block6 = BasicBlock(in_channels = 256, stride = 1)
        self.block7 = BasicBlock(in_channels = 256, stride = 2)
        self.block8 = BasicBlock(in_channels = 512, stride = 1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode = "fan_out", nonlinearity = "relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.bn(outputs)
        outputs = self.relu(outputs)
        # outputs = self.pool(outputs)

        outputs = self.block1(outputs)
        outputs = self.block2(outputs)
        outputs = self.block3(outputs)
        outputs = self.block4(outputs)
        outputs = self.block5(outputs)
        outputs = self.block6(outputs)
        outputs = self.block7(outputs)
        outputs = self.block8(outputs)

        outputs = self.pool(outputs)
        outputs = torch.flatten(outputs, 1)
        outputs = self.fc(outputs)

        return outputs


if __name__ == "__main__":
    print(get_num_parameters(ResNet(num_classes = 100)))