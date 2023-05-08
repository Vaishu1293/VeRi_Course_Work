from torch import nn
import torch.utils.model_zoo as model_zoo

__all__ = [
    "alexnet"
]

model_urls = {
    "alexnet": "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth"
}

class AlexNet(nn.Module):
    def __init__(self, num_classes, fc_dims=None, dropout_p=None, **kwargs):
        super(AlexNet, self).__init__()

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

        if fc_dims is not None:
            classifier_layers = []
            input_dim = 256 * 6 * 6
            for dim in fc_dims:
                classifier_layers.append(nn.Linear(input_dim, dim))
                classifier_layers.append(nn.ReLU(inplace=True))
                if dropout_p is not None:
                    classifier_layers.append(nn.Dropout(p=dropout_p))
                input_dim = dim
            self.fc = nn.Sequential(*classifier_layers)
            self.classifier = nn.Linear(fc_dims[-1], num_classes)
        else:
            self.fc = None
            self.classifier = nn.Linear(256 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.fc is not None:
            features = self.fc(x)
        else:
            features = x
        if not self.training:
            return x
        outputs = self.classifier(features)
        return outputs, features

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

def alexnet(num_classes, loss={"xent"}, pretrained=True, fc_dims=None, dropout_p=None, **kwargs):
    model = AlexNet(
        num_classes=num_classes,
        fc_dims=fc_dims,
        dropout_p=dropout_p,
        **kwargs,
    )
    if pretrained:
        init_pretrained_weights(model, model_urls["alexnet"])
    return model

