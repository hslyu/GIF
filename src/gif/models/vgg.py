"""VGG11/13/16/19 in Pytorch."""
import torch
import torch.nn as nn

cfg = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    def __init__(
        self,
        vgg_name,
        in_channels=3,
        num_classes=10,
        classifier_hidden_dim=512,
    ):
        super().__init__()
        self.features = self._make_layers(cfg[vgg_name], in_channels)
        self.classifier = nn.Sequential(
            nn.Linear(512, classifier_hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, in_channels):
        layers = []
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=False),
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGG16(in_channels=3, num_classes=10, classifier_hidden_dim=512):
    return VGG(
        "VGG16",
        in_channels=in_channels,
        num_classes=num_classes,
        classifier_hidden_dim=classifier_hidden_dim,
    )


def VGG11(in_channels=3, num_classes=10, classifier_hidden_dim=512):
    return VGG(
        "VGG11",
        in_channels=in_channels,
        num_classes=num_classes,
        classifier_hidden_dim=classifier_hidden_dim,
    )
