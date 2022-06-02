# 计算机视觉期末课程项目

GitHub repo 链接：<https://github.com/quniLcs/cv-final>

网盘链接：(百度网盘 计算机视觉期末作业)[<https://pan.baidu.com/s/1XgaH2ZibV2PRo8FYxgqymw?pwd=9qnk>]

## 语义分割任务

基于DeeplabV3，调用PixelLib库实现视频语义分割任务。

运行`Semantic segmentation.ipynb`文件，原视频及分割结果见百度网盘文件。

## 目标检测任务

### 随机初始化

使用VOC2007数据集进行训练和测试，将数据集解压至`VOCdevkit`文件夹。
	
修改代码文件：
`utils/config.py`修改参数`voc_data_dir`为数据集对应路径；修改预训练模型路径以加载随机初始化参数。
	
`model/faster_rcnn_vgg16.py`修改`line20`为`model = vgg16(pretrained=False)`

运行`model_train.ipynb`中`train()`函数开始训练。

### ImageNet预训练

使用VOC2007数据集进行训练和测试，将数据集解压至`VOCdevkit`文件夹。
	
修改代码文件：
`utils/config.py`修改参数`voc_data_dir`为数据集对应路径；修改预训练模型路径以加载ImageNet预训练参数。
	
`model/faster_rcnn_vgg16.py`修改`line20`为`model = vgg16(pretrained=True)`

运行`model_train.ipynb中train()`函数开始训练。

### COCO预训练Mask R-CNN

使用VOC2012数据集进行训练和测试，将数据集解压至`VOCdevkit`文件夹。
	
预训练参数设置：
运行`参数读取.ipynb`修改参数，Faster R-CNN模型的backbone参数部分替换为COCO数据集上预训练的Mask R-CNN模型的backbone参数，head部分使用numpy进行随机初始化。

运行`python train_res50_fpn.py`命令开始训练。

## 图像分类任务

### 使用模块说明

`argparse`：用于从命令行设置超参数；

`numpy`：用于数学计算；

`torch`：用于搭建并训练网络；

`torchvision`：用于加载数据集；

`vit_pytorch`：用于搭建网络框架；

`matplotlib.pyplot`：用于可视化。

### 代码文件说明

`load.py`：定义一个函数`load`，输入五个参数，第一个参数表示数据集文件路径，第二个参数表示是否训练集，第三个参数表示批量大小，第四个参数表示是否打乱顺序，第五个参数表示子进程数量；直接运行该文件时，调用该函数，保存训练集的前三张样本。

`cutout.py`：定义一个函数`cutout`，输入三个参数，第一个参数表示一批图像，第二个参数表示正方形边长，第三个参数表示硬件；直接运行该文件时，调用`load`加载数据集，并调用该函数，保存处理后的前三张样本。

`mixup.py`：定义一个函数`mixup`，输入三个参数，第一个参数表示一批图像，第二个参数表示一批标签，第三个参数表示Beta分布的参数；直接运行该文件时，调用`load`加载数据集，并调用该函数，保存处理后的前三张样本。

`cutmix.py`：定义一个函数`cutmix`，输入三个参数，含义与函数`mixup`相同；直接运行该文件时，调用`load`加载数据集，并调用该函数，保存处理后的前三张样本。

`util.py`：定义五个函数，`get_num_parameters`用于计算模型的参数数量；`optimize`用于使用指定的数据增强方法训练一个回合的模型，`evaluate`用于在训练集或测试集上评估模型，`save_status`用于保存模型和优化器，`load_status`用于加载模型。

`resnet.py`：定义模型`ResNet`；直接运行该文件时，计算模型的参数数量。

`transformer.py`：定义模型`Transformer`；直接运行该文件时，计算模型的参数数量。

`main.py`：调用`load`加载训练集和测试集，实例化`ResNet`或`Transformer`，学习率阶梯下降且带有动量的随机梯度下降优化器、交叉熵损失函数、调用`optimize`和`evaluate`训练并测试模型，最后调用`save_status`保存模型和优化器。

### 训练和测试示例代码

在命令行训练不同的模型：

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python main.py --model resnet
python main.py --model transformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

其它命令行超参数：

`batch_size`：表示批量大小，默认值`128`；

`num_epoch`：表示回合数，默认值`80`；

`lr`：表示初始学习率，默认值`0.1`；

`milestones`：表示学习率下降回合列表，默认值`[20, 40, 60]`；

`gamma`：表示学习率下降参数，默认值`0.2`；

`momentum`：表示动量，默认值`0.9`；

`lambd`：表示正则化参数，默认值`5e-4`；

`mode`：表示数据增强方法，默认值`baseline`，可改为`cutout`，`mixup`，`cutmix`。
