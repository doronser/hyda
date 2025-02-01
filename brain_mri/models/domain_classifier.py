from torch import nn

class GideonsEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # block 1
            nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.BatchNorm3d(8),

            # block 2
            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.BatchNorm3d(16),

            # block 3
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.BatchNorm3d(32),

            # block 4
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.BatchNorm3d(64),
        )

    def forward(self, x):
        return self.model(x)

class GideonsMLP(nn.Module):
    def __init__(self, in_features, num_classes, dropout=0.1):
        super().__init__()

        self.in_features = in_features
        self.num_classes = num_classes
        self.dropout = dropout
        self.model = nn.Sequential(
            # linear block
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(in_features, 32),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.model(x)