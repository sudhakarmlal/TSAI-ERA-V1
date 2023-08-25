"""Implementation of YOLOv3 architecture."""
from typing import Any, List

import torch
import torch.nn as nn
import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
from torch.optim.lr_scheduler import OneCycleLR


"""
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride)
Every conv is a same convolution.
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )


class YOLOv3(nn.Module):
    def __init__(self, load_config: List[Any] = config, in_channels=3, num_classes=80):
        super().__init__()
        self.load_config = load_config
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []  # for each scale
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in self.load_config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(
                    ResidualBlock(
                        in_channels,
                        num_repeats=num_repeats,
                    )
                )

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(
                        nn.Upsample(scale_factor=2),
                    )
                    in_channels = in_channels * 3

        return layers


class Assignment13(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.epoch_number = 0
        self.config = config
        self.train_csv_path = self.config.DATASET + "/train.csv"
        self.test_csv_path = self.config.DATASET + "/test.csv"
        self.train_loader, self.test_loader, self.train_eval_loader = get_loaders(
              train_csv_path=self.train_csv_path, test_csv_path=self.test_csv_path)
        self.check_class_accuracy = check_class_accuracy
        self.model = YOLOv3(num_classes=self.config.NUM_CLASSES)
        self.loss_fn = YoloLoss()
        self.check_class_accuracy = check_class_accuracy
        self.get_evaluation_bboxes = get_evaluation_bboxes
        self.scaled_anchors = (torch.tensor(self.config.ANCHORS) * torch.tensor(self.config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2))
        self.losses = []
        self.plot_couple_examples = plot_couple_examples
        self.mean_average_precision = mean_average_precision
        self.EPOCHS = self.config.NUM_EPOCHS * 2 // 5
    def forward(self, x):
        out = self.model(x)
        return out
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        y0, y1, y2 = (y[0],y[1],y[2])
        loss = (
                self.loss_fn(out[0], y0, self.scaled_anchors[0].to(y0))
                + self.loss_fn(out[1], y1, self.scaled_anchors[1].to(y1))
                + self.loss_fn(out[2], y2, self.scaled_anchors[2].to(y2))
            )
        self.losses.append(loss.item())
        mean_loss = sum(self.losses) / len(self.losses)
        self.log("train_loss", mean_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log("train_loss", mean_loss)
        return loss


    def on_train_epoch_start(self):
        self.epoch_number += 1
        self.losses = []
        #self.plot_couple_examples(self.model,self.test_loader,0.6,0.5,self.scaled_anchors)
        if self.epoch_number > 1 and self.epoch_number % 10 == 0:
            self.plot_couple_examples(self.model,self.test_loader,0.6,0.5,self.scaled_anchors)

    def on_train_epoch_end(self):
        print(f"Currently epoch {self.epoch_number}")
        print("On Train Eval loader:")
        print("On Train loader:")
        self.check_class_accuracy(self.model, self.train_loader, threshold=self.config.CONF_THRESHOLD)
        if self.epoch_number == self.EPOCHS:
              #if self.epoch_number > 1 and self.epoch_number % 3 == 0:
            self.check_class_accuracy(self.model, self.test_loader, threshold=self.config.CONF_THRESHOLD)
            pred_boxes, true_boxes = self.get_evaluation_bboxes( self.test_loader,self.model,iou_threshold=self.config.NMS_IOU_THRESH,
                                                                 anchors=self.config.ANCHORS,
                                                                 threshold=self.config.CONF_THRESHOLD,)
            mapval = self.mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=self.config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=self.config.NUM_CLASSES,
            )
            print(f"MAP: {mapval.item()}")
            self.model.train()
            pass


    def configure_optimizers(self):
        optimizer = optimizer = optim.Adam(
                    model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        #EPOCHS = config.NUM_EPOCHS * 2 // 5
        scheduler = OneCycleLR(
        optimizer,
        max_lr=1E-3,
        steps_per_epoch=len(self.train_loader),
        epochs=self.EPOCHS,
        pct_start=5/self.EPOCHS,
        div_factor=100,
        three_phase=False,
        final_div_factor=100,
        anneal_strategy='linear'
        )

        return {"optimizer": optimizer, "lr_scheduler":scheduler}

     ####################
    # DATA RELATED HOOKS
    ####################

    def train_dataloader(self):
          return self.train_loader

    def test_dataloader(self):
          return self.test_loader
                        
if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLOv3(load_config=config, num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    assert out[0].shape == (2, 3, IMAGE_SIZE // 32, IMAGE_SIZE // 32, num_classes + 5)
    assert out[1].shape == (2, 3, IMAGE_SIZE // 16, IMAGE_SIZE // 16, num_classes + 5)
    assert out[2].shape == (2, 3, IMAGE_SIZE // 8, IMAGE_SIZE // 8, num_classes + 5)
    print("Success!")
