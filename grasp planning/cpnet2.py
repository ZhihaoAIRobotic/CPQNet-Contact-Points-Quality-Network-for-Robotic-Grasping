# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F

filter_sizes = [16, 8, 8, 8, 8, 16]
kernel_sizes = [9, 5, 3, 3, 5, 9]
strides = [3, 2, 2, 2, 2, 3]


class CPNET(nn.Module):
    def __init__(self, input_channels=1):
        super(CPNET, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, filter_sizes[0], kernel_sizes[0], stride=strides[0], padding=3)
        self.conv2 = nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_sizes[1], stride=strides[1], padding=2)
        self.conv3 = nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_sizes[2], stride=strides[2], padding=1)

        self.convt1 = nn.ConvTranspose2d(filter_sizes[2], filter_sizes[3], kernel_sizes[3], stride=strides[3],
                                         padding=1, output_padding=1)
        self.convt2 = nn.ConvTranspose2d(filter_sizes[3], filter_sizes[4], kernel_sizes[4], stride=strides[4],
                                         padding=2, output_padding=1)
        self.convt3 = nn.ConvTranspose2d(filter_sizes[4], filter_sizes[5], kernel_sizes[5], stride=strides[5],
                                         padding=3, output_padding=1)

        self.pos_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.convt1(x))
        x = F.relu(self.convt2(x))
        x = F.relu(self.convt3(x))

        pos_output = self.pos_output(x)

        return pos_output

    def compute_loss(self, xc, yc):
        y_pos = yc
        pos_pred = self(xc)
        p_loss = F.mse_loss(pos_pred, y_pos, reduction='sum')
        return {
            'loss':  p_loss,
            'pred': {
                'pos': pos_pred
            }
        }



