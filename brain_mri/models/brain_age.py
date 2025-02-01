from torch import nn


class AgeEncoder(nn.Module):
    def __init__(self,
                 layers=(16, 32, 64),
                 kernel_size=3,
                 norm_layer='BatchNorm3d',
                 activation_layer='LeakyReLU'
                 ):
        super().__init__()
        norm_layer = getattr(nn, norm_layer)
        activation = getattr(nn, activation_layer)
        self.model = nn.Sequential()
        for i, (in_c, out_c) in enumerate(zip([1] + list(layers), layers)):
            self.model.add_module(f"conv_{in_c}_{out_c}",
                                  nn.Conv3d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size))
            self.model.add_module(f"norm_{i}", norm_layer(num_features=out_c))
            self.model.add_module(f"activation_{i}", activation())
            if i < len(layers) - 1:
                self.model.add_module(f"pool_{i}", nn.MaxPool3d(kernel_size=2))
        self.model.add_module("avg_pool", nn.AdaptiveAvgPool3d((2, 2, 2)))

    def forward(self, x):
        return self.model(x)

class AgeRegressor(nn.Module):
    def __init__(self,
                 in_features,
                 num_classes=1,
                 layers=(1024, 256, 64),
                 dropout_probability=0,
                 activation_layer='LeakyReLU'
                 ):
        super().__init__()
        activation = getattr(nn, activation_layer)

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_features, out_features=layers[0]),  # orig 74880
            activation(),
            nn.Dropout(p=dropout_probability),
            nn.Linear(in_features=layers[0], out_features=layers[1]),
            activation(),
            nn.Linear(in_features=layers[1], out_features=layers[2]),
            activation(),
            nn.Linear(in_features=layers[2], out_features=num_classes),
        )

    def forward(self, x):
        return self.model(x)

class Model(nn.Module):
    def __init__(self,
                 dropout_probability=0,
                 out_channels=1,
                 kernel_size=3,
                 norm_layer='BatchNorm3d',
                 activation_layer='LeakyReLU',
                 encoder_layers=(16, 32, 64),
                 regressor_layers=(1024, 256, 64),
                 ):
        super().__init__()
        self.features = AgeEncoder(layers=encoder_layers, kernel_size=kernel_size,
                                   norm_layer=norm_layer, activation_layer=activation_layer)
        self.regressor = AgeRegressor(in_features=8*encoder_layers[-1], num_classes=out_channels,
                                      layers=regressor_layers, dropout_probability=dropout_probability,
                                      activation_layer=activation_layer)

    def forward(self, scan):
        age_prediction = self.regressor(self.features(scan))
        return age_prediction