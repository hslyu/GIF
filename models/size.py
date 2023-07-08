"""Train CIFAR10 with PyTorch."""

from densenet import *
from dla import *
from dla_simple import *
from dpn import *
from efficientnet import *
from fcn import *
from googlenet import *
from lenet import *
from mobilenet import *
from mobilenetv2 import *
from pnasnet import *
from preact_resnet import *
from regnet import *
from resnet import *
from resnext import *
from senet import *
from shufflenet import *
from shufflenetv2 import *
from tiny import *
from vgg import *

net_list = [
    VGG("VGG16"),
    ResNet18(),
    PreActResNet18(),
    GoogLeNet(),
    DenseNet121(),
    ResNeXt29_2x64d(),
    MobileNet(),
    MobileNetV2(),
    DPN92(),
    # ShuffleNetG2(),
    SENet18(),
    ShuffleNetV2(1),
    EfficientNetB0(),
    RegNetX_200MF(),
    TinyNet(),
    SimpleDLA(),
    DenseNet121(),
]
for net in net_list:
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Network: {net.__class__.__name__}, Parameters={num_params/1000000:.2f}M")
