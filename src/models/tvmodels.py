# Copyright (c) EEEM071, University of Surrey

import torch.nn as nn
import torchvision.models as tvmodels


__all__ = ["mobilenet_v3_small", "vgg16", "alexnet", "densenet121", "densenet169", "densenet201"]


import torch.nn.functional as F

class TorchVisionModel(nn.Module):
    def __init__(self, name, num_classes, loss, pretrained, **kwargs):
        super().__init__()

        self.loss = loss
        self.backbone = tvmodels.__dict__[name](pretrained=pretrained)

        if name == "alexnet":
            # Modify the last layer to have the desired number of output features
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
            self.feature_dim = in_features

            # Add an AdaptiveAvgPool2d layer to reshape the penultimate layer output
            self.pool = nn.AdaptiveAvgPool2d((6, 6))

            # Modify the classifier to have the desired input dimensions
            self.classifier = nn.Linear(in_features * 6 * 6, num_classes)
        else:
            # Modify the last layer to have the desired number of output features
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
            self.feature_dim = in_features

        # overwrite the classifier used for ImageNet pretrianing
        # nn.Identity() will do nothing, it's just a place-holder
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        v = self.backbone(x)

        if not self.training:
            return v

        if len(v.shape) < 3:
            # Add a dimension to the tensor to make it at least 3D
            v = v.unsqueeze(2)

        v = self.pool(v)
        v = v.view(v.size(0), -1)

        y = self.classifier(v)

        if self.loss == {"xent"}:
            return y
        elif self.loss == {"xent", "htri"}:
            return y, v
        else:
            raise KeyError(f"Unsupported loss: {self.loss}")



def vgg16(num_classes, loss={"xent"}, pretrained=True, **kwargs):
    model = TorchVisionModel(
        "vgg16",
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        **kwargs,
    )
    return model


def mobilenet_v3_small(num_classes, loss={"xent"}, pretrained=True, **kwargs):
    model = TorchVisionModel(
        "mobilenet_v3_small",
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        **kwargs,
    )
    return model
    
def alexnet(num_classes, loss={"xent"}, pretrained=True, **kwargs):
    model = TorchVisionModel(
        "alexnet",
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        **kwargs,
    )
    return model

def densenet121(num_classes, loss={"xent"}, pretrained=True, **kwargs):
    model = TorchVisionModel(
        "densenet121",
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        **kwargs,
    )
    return model


def densenet169(num_classes, loss={"xent"}, pretrained=True, **kwargs):
    model = TorchVisionModel(
        "densenet169",
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        **kwargs,
    )
    return model


def densenet201(num_classes, loss={"xent"}, pretrained=True, **kwargs):
    model = TorchVisionModel(
        "densenet201",
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        **kwargs,
    )
    return model



# Define any models supported by torchvision bellow
# https://pytorch.org/vision/0.11/models.html
