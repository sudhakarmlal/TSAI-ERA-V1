import torch.nn as nn
import torchinfo
    

class ConvLayer(nn.Module):
    def __init__(self, input_channels, output_channels, bias=False, padding=0, dws=False, skip=False, dilation=1, dropout=0):
        super(ConvLayer, self).__init__()

        # Member Variables
        self.skip = skip

        # If Depthwise Separable is True
        if dws and input_channels == output_channels:
            self.convlayer = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 3, bias=bias, padding=padding, groups=input_channels, dilation=dilation,
                          padding_mode='replicate'),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(),
                nn.Conv2d(output_channels, output_channels, 1, bias=bias)
            )
        else:
            self.convlayer = nn.Conv2d(input_channels, output_channels, 3, bias=bias, padding=padding, groups=1, dilation=dilation,
                                       padding_mode='replicate')

        self.norm = nn.BatchNorm2d(output_channels)

        self.skiplayer = None
        if self.skip and input_channels != output_channels:
            self.skiplayer = nn.Conv2d(input_channels, output_channels, 1, bias=bias)

        self.activation = nn.ReLU()

        self.droplayer = None
        if dropout > 0:
            self.droplayer = nn.Dropout(dropout)

    def forward(self, x):
        x_ = x
        x = self.convlayer(x)
        x = self.norm(x)
        if self.skip:
            if self.skiplayer is None:
                x += x_
            else:
                x += self.skiplayer(x_)
        x = self.activation(x)
        if self.droplayer is not None:
            x = self.droplayer(x)
        return x


class Model(nn.Module):
    def __init__(self, dropout=0, skip=False):
        super(Model, self).__init__()

        self.dropout = dropout

        self.conv1 = self.get_conv_block(input_channels= 3, output_channels= 16, padding=1, dws=True, skip=False, reps=2, dropout=self.dropout)
        self.t1 = self.get_trans_block(input_channels= 16, output_channels= 32, padding=0, dws=False, skip=False, dilation=1, dropout=self.dropout)
        self.conv2 = self.get_conv_block(input_channels= 32, output_channels= 32, padding=1, dws=True, skip=skip, reps=2, dropout=self.dropout)
        self.t2 = self.get_trans_block(input_channels= 32, output_channels= 64, padding=0, dws=False, skip=False, dilation=2, dropout=self.dropout)
        self.conv3 = self.get_conv_block(input_channels= 64, output_channels= 64, padding=1, dws=True, skip=skip, reps=2, dropout=self.dropout)
        self.t3 = self.get_trans_block(input_channels= 64, output_channels= 96, padding=0, dws=False, skip=False, dilation=4, dropout=self.dropout)
        self.conv4 = self.get_conv_block(input_channels= 96, output_channels= 96, padding=1, dws=True, skip=skip, reps=2, dropout=self.dropout)
        self.t4 = self.get_trans_block(input_channels= 96, output_channels= 96, padding=0, dws=False, skip=False, dilation=8, dropout=0)

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(96, 10, 1, bias=True),
            nn.Flatten(),
            nn.LogSoftmax(-1)
        )

    @staticmethod
    def get_conv_block(input_channels, output_channels, bias=False, padding=0, dws=True, skip=True, reps=2, dilation=1,
                       dropout=0):
        block = list()
        for i in range(0, reps):
            block.append(
                ConvLayer(output_channels if i > 0 else input_channels, output_channels, bias=bias, padding=padding, dws=dws, skip=skip,
                          dilation=dilation, dropout=dropout)
            )
        return nn.Sequential(*block)

    @staticmethod
    def get_trans_block(input_channels, output_channels, bias=False, padding=0, dws=False, skip=False, dilation=1, dropout=0):
        return ConvLayer(input_channels, output_channels, bias=bias, padding=padding, dws=dws, skip=skip, dilation=dilation,
                         dropout=dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.t1(x)
        x = self.conv2(x)
        x = self.t2(x)
        x = self.conv3(x)
        x = self.t3(x)
        x = self.conv4(x)
        x = self.t4(x)
        x = self.gap(x)
        return x

    
    def summary(self, input_size=None):
        return torchinfo.summary(self, input_size=input_size, depth=5,
                                 col_names=["input_size", "output_size", "num_params", "params_percent"])
                    


                                        