# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def SyncBatchNorm(*args, **kwargs):
    """In cpu environment nn.SyncBatchNorm does not have kernel so use nn.BatchNorm2D instead"""
    if paddle.get_device() == 'cpu' or os.environ.get('PADDLESEG_EXPORT_STAGE'):
        return nn.BatchNorm2D(*args, **kwargs)
    elif paddle.distributed.ParallelEnv().nranks == 1:
        return nn.BatchNorm2D(*args, **kwargs)
    else:
        return nn.SyncBatchNorm(*args, **kwargs)


class ConvBNReLU(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2D(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)

        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self._batch_norm = SyncBatchNorm(out_channels, data_format=data_format)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = F.relu(x)
        return x


class ConvBN(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()
        self._conv = nn.Conv2D(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)
        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self._batch_norm = SyncBatchNorm(out_channels, data_format=data_format)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        return x


class ConvReLUPool(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = F.pool2d(x, pool_size=2, pool_type="max", pool_stride=2)
        return x


class SeparableConvBNReLU(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()
        self.depthwise_conv = ConvBN(
            in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            **kwargs)
        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self.piontwise_conv = ConvBNReLU(
            in_channels,
            out_channels,
            kernel_size=1,
            groups=1,
            data_format=data_format)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.piontwise_conv(x)
        return x


class DepthwiseConvBN(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()
        self.depthwise_conv = ConvBN(
            in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            **kwargs)

    def forward(self, x):
        x = self.depthwise_conv(x)
        return x


class AuxLayer(nn.Layer):
    """
    The auxiliary layer implementation for auxiliary loss.

    Args:
        in_channels (int): The number of input channels.
        inter_channels (int): The intermediate channels.
        out_channels (int): The number of output channels, and usually it is num_classes.
        dropout_prob (float, optional): The drop rate. Default: 0.1.
    """

    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 dropout_prob=0.1):
        super().__init__()

        self.conv_bn_relu = ConvBNReLU(
            in_channels=in_channels,
            out_channels=inter_channels,
            kernel_size=3,
            padding=1)

        self.dropout = nn.Dropout(p=dropout_prob)

        self.conv = nn.Conv2D(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=1)

    def forward(self, x):
        x = self.conv_bn_relu(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x


class ConvBNPReLU(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 groups=1):
        super().__init__()

        self._conv = nn.Conv2D(in_channels, out_channels, kernel_size, padding='same', stride=stride, groups=groups, bias_attr=False)
        self._bn = SyncBatchNorm(out_channels)
        self._act = nn.PReLU()

    def forward(self, x):
        x = self._conv(x)
        x = self._bn(x)
        x = self._act(x)
        return x


class ConvBN(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1):
        super().__init__()
        self._conv = nn.Conv2D(in_channels, out_channels, kernel_size, padding='same', stride=stride, groups=groups, bias_attr=False)
        self._bn = SyncBatchNorm(out_channels)

    def forward(self, x):
        x = self._conv(x)
        x = self._bn(x)
        return x


class BNPReLU(nn.Layer):
    def __init__(self, out_channels):
        super().__init__()
        self._bn = SyncBatchNorm(out_channels)
        self._act = nn.PReLU()

    def forward(self, x):
        x = self._bn(x)
        x = self._act(x)
        return x


class EESP(nn.Layer):
    """
    EESP block, principle: reduce -> split -> transform -> merge

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Factor by which we should skip (useful for down-sampling). If 2, then down-samples the feature map by 2.
        branches (int): Number of branches.
        kernel_size_maximum (int): A maximum value of receptive field allowed for EESP block.
        down_method (str): Down sample or not, only support 'avg' and 'esp'. (equivalent to stride is 2 or not)
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 branches=4,
                 kernel_size_maximum=7,
                 down_method='esp'):
        super(EESP, self).__init__()
        assert out_channels % branches == 0, "out_channels should be a multiple of branches."
        assert down_method in ['avg', 'esp'], "down_method only support 'avg' or 'esp'."
        self.in_channels = in_channels
        self.stride = stride

        in_branch_channels = int(out_channels / branches)
        self.group_conv_in = ConvBNPReLU(in_channels, in_branch_channels, 1, stride=1, groups=branches)

        map_ksize_dilation = {3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7, 17: 8}
        self.kernel_sizes = []
        for i in range(branches):
            kernel_size = 3 + 2 * i
            kernel_size = kernel_size if kernel_size <= kernel_size_maximum else 3
            self.kernel_sizes.append(kernel_size)
        self.kernel_sizes.sort()

        self.spp_modules = nn.LayerList()
        for i in range(branches):
            dilation = map_ksize_dilation[self.kernel_sizes[i]]
            self.spp_modules.append(
                nn.Conv2D(in_branch_channels,
                          in_branch_channels,
                          kernel_size=3,
                          padding='same',
                          stride=stride,
                          dilation=dilation,
                          groups=in_branch_channels,
                          bias_attr=False)
            )
        self.group_conv_out = ConvBN(out_channels, out_channels, kernel_size=1, stride=1, groups=branches)
        self.bn_act = BNPReLU(out_channels)
        self._act = nn.PReLU()
        self.downAvg = True if down_method == 'avg' else False

    def forward(self, x):
        group_out = self.group_conv_in(x)  # reduce feature map dimensions
        output = [self.spp_modules[0](group_out)]

        # compute the output for each branch and hierarchically fuse them
        for k in range(1, len(self.spp_modules)):
            output_k = self.spp_modules[k](group_out)
            output_k = output_k + output[k - 1]     # HFF
            output.append(output_k)

        group_merge = self.group_conv_out(self.bn_act(paddle.concat(output, axis=1)))   # merge
        del output

        # if down-sampling, then return the merged feature map.
        if self.stride == 2 and self.downAvg:
            return group_merge

        # residual link
        if group_merge.shape == x.shape:
            group_merge = group_merge + x
        return self._act(group_merge)

