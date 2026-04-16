"""Parameter count summary for bundled model definitions."""

from . import (
    DPN92,
    DenseNet121,
    EfficientNetB0,
    GoogLeNet,
    MobileNet,
    MobileNetV2,
    PreActResNet18,
    RegNetX_200MF,
    ResNeXt29_2x64d,
    ResNet18,
    SENet18,
    ShuffleNetV2,
    SimpleDLA,
    TinyNet,
    VGG,
)


def build_model_list():
    return [
        VGG("VGG16"),
        ResNet18(),
        PreActResNet18(),
        GoogLeNet(),
        DenseNet121(),
        ResNeXt29_2x64d(),
        MobileNet(),
        MobileNetV2(),
        DPN92(),
        SENet18(),
        ShuffleNetV2(1),
        EfficientNetB0(),
        RegNetX_200MF(),
        TinyNet(),
        SimpleDLA(),
        DenseNet121(),
    ]


def main():
    for net in build_model_list():
        num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"Network: {net.__class__.__name__}, Parameters={num_params/1000000:.2f}M")


if __name__ == "__main__":
    main()
