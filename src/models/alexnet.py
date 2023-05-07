import torch
import torch.utils.model_zoo as model_zoo
import torchvision
from torch import nn
from torch.nn import functional as F

__all__ = [
    "alexnet",
    "alexnet_fc512",
]

model_urls = {
    "alexnet": "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth",
    "alexnet_fc512": "https://github.com/mf1024/Image-Retrieval-with-Deep-Local-Features/releases/download/v1.0/alexnet_fc512.pth",
}

import torch
import torch.utils.model_zoo as model_zoo
import torchvision
from torch import nn
from torch.nn import functional as F

__all__ = [
    "alexnet",
    "alexnet_fc512",
]

model_urls = {
    "alexnet": "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth",
    "alexnet_fc512": "https://github.com/mf1024/Image-Retrieval-with-Deep-Local-Features/releases/download/v1.0/alexnet_fc512.pth",
}

class AlexNet(nn.Module):
    """
    AlexNet implementation
    """

    def __init__(self, num_classes, loss={"xent"}, pretrained=True, **kwargs):
        super().__init__()
        self.loss = loss
        self.feature_dim = 512
        
        # define network layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
        )

        self.classifier_last = nn.Linear(self.feature_dim, num_classes)

        if pretrained:
            self.init_pretrained_weights()

    def init_pretrained_weights(self):
        """
        Initialize model with pretrained weights.
        Layers that don't match with pretrained layers in name or size are kept unchanged.
        """
        pretrain_dict = model_zoo.load_url(model_urls["alexnet"])
        model_dict = self.state_dict()
        pretrain_dict = {
            k: v
            for k, v in pretrain_dict.items()
            if k in model_dict and model_dict[k].size() == v.size()
        }
        model_dict.update(pretrain_dict)
        self.load_state_dict(model_dict)
        print(f"Initialized model with pretrained weights from {model_urls['alexnet']}")

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        if not self.training:
            return x

        x = self.classifier_last(x)

        if self.loss == {"xent"}:
            return x
        elif self.loss == {"xent", "htri"}:
            return x, None
        else:
            raise KeyError(f"Unsupported loss: {self.loss}")


def init_pretrained_weights(model, model_url):
    """
    Initialize model with pretrained weights.
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {
        k: v
        for k, v in pretrain_dict.items()
        if k in model_dict and model_dict[k].size() == v.size()
    }
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    print(f"Initialized model with pretrained weights from {model_url}")

def alexnet(num_classes, loss={"xent"}, pretrained=True, **kwargs):
    model = AlexNet(num_classes=num_classes, loss=loss, pretrained=pretrained, **kwargs)
    return model


def alexnet_fc512(num_classes, loss={"xent"}, pretrained=True, **kwargs):
    model = AlexNet(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        fc_dims=[512],
        **kwargs,
    )
    return model





