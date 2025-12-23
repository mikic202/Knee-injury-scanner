from torch import nn


class CnnKneeClassifier(nn.Module):
    def __init__(self, input_channels: int, num_classes: int):
        super(CnnKneeClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.last_feature = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.linear_layers = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8 * 8, num_classes),
            # nn.ReLU(),
            # nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.last_feature(x)
        x = self.linear_layers(x)
        return x
