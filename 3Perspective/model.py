import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, name) -> None:
        super().__init__()
        self.model = torch.hub.load(
            'pytorch/vision:v0.10.0', name, pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

        # model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet169', pretrained=True)

        # model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)

        # model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)

        self.score = nn.Sequential(
            nn.ReLU(),
            nn.LazyLinear(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.model(x)
        x = self.score(x)
        return x
