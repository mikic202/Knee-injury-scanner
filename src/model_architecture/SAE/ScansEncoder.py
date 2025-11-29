from torch import nn


class ScansEncoder3d(nn.Module):
    def __init__(self, input_channels=1, feature_dim=256):
        super(ScansEncoder3d, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
        )
        self.fc = nn.Linear(128 * 8 * 8 * 8, feature_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ScansDecoder3d(nn.Module):
    def __init__(self, output_channels=1, feature_dim=256):
        super(ScansDecoder3d, self).__init__()
        self.fc = nn.Linear(feature_dim, 128 * 8 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(64, output_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 128, 8, 8, 8)
        x = self.decoder(x)
        return x


class ScansAutoencoder3d(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, feature_dim=256):
        super(ScansAutoencoder3d, self).__init__()
        self.encoder = ScansEncoder3d(input_channels, feature_dim)
        self.decoder = ScansDecoder3d(output_channels, feature_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = nn.functional.interpolate(
            decoded, size=x.shape[2:], mode="trilinear", align_corners=False
        )
        return decoded
