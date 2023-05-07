import torch.nn as nn
import torchvision.models as tvmodels


__all__ = ["mobilenet_v3_small", "vgg16", "squeezenet"]


class TorchVisionModel(nn.Module):
    def __init__(self, name, num_classes, loss, pretrained, **kwargs):
        super().__init__()

        self.loss = loss
        self.backbone = tvmodels.__dict__[name](pretrained=pretrained)

        if name == "vgg16":
            self.feature_dim = self.backbone.classifier[0].in_features
        elif name == "squeezenet1_0":
            self.feature_dim = 512
            self.backbone.num_classes = num_classes
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Conv2d(512, num_classes, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=13, stride=1)
            )
        else:
            self.feature_dim = self.backbone.classifier[-1].in_features

        # overwrite the classifier used for ImageNet pretraining
        # nn.Identity() will do nothing, it's just a placeholder
        self.backbone.classifier = nn.Identity()
        if name != "squeezenet1_0":
            self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        v = self.backbone(x)

        if not self.training:
            return v

        if len(v.shape) < 3:
            # Add a dimension to the tensor to make it at least 3D
            v = v.unsqueeze(2)

        if isinstance(self.classifier, nn.Linear):
            v = v.mean(dim=2)  # Global average pooling
            y = self.classifier(v)
        else:
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


def squeezenet(num_classes, loss={"xent"}, pretrained=True, **kwargs):
    model = TorchVisionModel(
        "squeezenet1_0",
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        **kwargs,
    )
    return model


# Define any models supported by torchvision below
# https://pytorch.org/vision/0.11/models.html
