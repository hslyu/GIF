from .densenet import DenseNet121, DenseNet169, DenseNet201, DenseNet161
from .dla import DLA
from .dla_simple import SimpleDLA
from .dpn import DPN26, DPN92
from .efficientnet import EfficientNetB0
from .fcn import FullyConnectedNet
from .googlenet import GoogLeNet
from .lenet import LeNet
from .lora import (
    LoRALinear,
    LoRAFullyConnectedNet,
    count_trainable_parameters,
    get_trainable_parameters,
    load_base_state_dict_into_lora,
    trainable_parameters_to_vector,
    vector_to_trainable_parameters,
)
from .mobilenet import MobileNet
from .mobilenetv2 import MobileNetV2
from .pnasnet import PNASNetA, PNASNetB
from .preact_resnet import (
    PreActResNet18,
    PreActResNet34,
    PreActResNet50,
    PreActResNet101,
    PreActResNet152,
)
from .regnet import RegNetX_200MF, RegNetX_400MF, RegNetY_400MF
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .resnext import ResNeXt29_2x64d, ResNeXt29_4x64d, ResNeXt29_8x64d, ResNeXt29_32x4d
from .senet import SENet18
from .shufflenet import ShuffleNetG2, ShuffleNetG3
from .shufflenetv2 import ShuffleNetV2
from .tiny import TinyNet
from .vgg import VGG, VGG11, VGG16

MODEL_REGISTRY = {
    "DLA": DLA,
    "DenseNet121": DenseNet121,
    "DenseNet161": DenseNet161,
    "DenseNet169": DenseNet169,
    "DenseNet201": DenseNet201,
    "DPN26": DPN26,
    "DPN92": DPN92,
    "EfficientNetB0": EfficientNetB0,
    "FullyConnectedNet": FullyConnectedNet,
    "GoogLeNet": GoogLeNet,
    "LeNet": LeNet,
    "MobileNet": MobileNet,
    "MobileNetV2": MobileNetV2,
    "PNASNetA": PNASNetA,
    "PNASNetB": PNASNetB,
    "PreActResNet18": PreActResNet18,
    "PreActResNet34": PreActResNet34,
    "PreActResNet50": PreActResNet50,
    "PreActResNet101": PreActResNet101,
    "PreActResNet152": PreActResNet152,
    "RegNetX_200MF": RegNetX_200MF,
    "RegNetX_400MF": RegNetX_400MF,
    "RegNetY_400MF": RegNetY_400MF,
    "ResNeXt29_2x64d": ResNeXt29_2x64d,
    "ResNeXt29_4x64d": ResNeXt29_4x64d,
    "ResNeXt29_8x64d": ResNeXt29_8x64d,
    "ResNeXt29_32x4d": ResNeXt29_32x4d,
    "ResNet18": ResNet18,
    "ResNet34": ResNet34,
    "ResNet50": ResNet50,
    "ResNet101": ResNet101,
    "ResNet152": ResNet152,
    "SENet18": SENet18,
    "ShuffleNetG2": ShuffleNetG2,
    "ShuffleNetG3": ShuffleNetG3,
    "ShuffleNetV2": ShuffleNetV2,
    "SimpleDLA": SimpleDLA,
    "TinyNet": TinyNet,
    "VGG": VGG,
    "VGG11": VGG11,
    "VGG16": VGG16,
}

__all__ = [
    "DLA",
    "DenseNet121",
    "DenseNet161",
    "DenseNet169",
    "DenseNet201",
    "DPN26",
    "DPN92",
    "EfficientNetB0",
    "FullyConnectedNet",
    "GoogLeNet",
    "LeNet",
    "LoRALinear",
    "LoRAFullyConnectedNet",
    "MODEL_REGISTRY",
    "MobileNet",
    "MobileNetV2",
    "PNASNetA",
    "PNASNetB",
    "PreActResNet18",
    "PreActResNet34",
    "PreActResNet50",
    "PreActResNet101",
    "PreActResNet152",
    "RegNetX_200MF",
    "RegNetX_400MF",
    "RegNetY_400MF",
    "ResNeXt29_2x64d",
    "ResNeXt29_4x64d",
    "ResNeXt29_8x64d",
    "ResNeXt29_32x4d",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
    "SENet18",
    "ShuffleNetG2",
    "ShuffleNetG3",
    "ShuffleNetV2",
    "SimpleDLA",
    "TinyNet",
    "VGG",
    "VGG11",
    "VGG16",
    "count_trainable_parameters",
    "get_trainable_parameters",
    "load_base_state_dict_into_lora",
    "trainable_parameters_to_vector",
    "vector_to_trainable_parameters",
]
