# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from layers import Conv3x3, ConvBlock


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        # layers         4     3     2    1    0
        # input_features
        # resolution     1/32  1/16  1/8  1/4  1/2
        # channel        512   256   128  64   32
        # outputs
        # resolution           1/8   1/4  1/2  1/1

        # monodepth2 decoder
        # 1/32,512    1/32,256   1/16,512    1/16,256   1/16,128    1/8,256     1/8,128     1/8,64     1/4,128     1/4,64      1/4,32     1/2,96      1/2,32      1/2,16     1/1,16      1/1,16
        #      upconv4_0   upsample   upconv4_1   upconv3_0   upsample   upconv3_1   upconv2_0   upsample   upconv2_1   upconv1_0   upsample   upconv1_1   upconv0_0   upsample   upconv0_1
        # x[4] --------> x -------> x --------> x --------> x -------> x --------> x --------> x -------> x --------> x --------> x -------> x --------> x --------> x -------> x --------> x
        #              x[3]--cat--↗                       x[2]--cat--↗    dispconv3↓         x[1]--cat--↗    dispconv2↓         x[0]--cat--↗    dispconv1↓                         dispconv1↓
        #            1/32,256                           1/16,128                 disp3      1/4,64                  disp2      1/2,64                  disp1                              disp0
        #                                                                        1/8,1                              1/4,1                              1/2,1                              1/1,1
        
        x = input_features[-1]  # 1/32
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [nn.functional.interpolate(x, scale_factor=2, mode="nearest")]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs
