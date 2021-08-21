import paddle
from paddle import nn
import paddle.nn.functional as F
from paddleseg.cvlibs import manager



class ConvBNLayer(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=1):
        super(ConvBNLayer, self).__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm2D(out_channels)
        self.act = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class CDilated(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, d=1, groups=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2) * d
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                              bias_attr=False, dilation=d, groups=groups)

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvBn(nn.Layer):
    """
    no activation
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2D(in_channels,out_channels, kernel_size, stride=stride, padding=padding,
                              bias_attr=False,groups=groups)
        self.bn = nn.BatchNorm2D(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class BnAct(nn.Layer):
    '''
        This class groups the batch normalization and PReLU activation
    '''

    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.bn = nn.BatchNorm2D(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        output = self.bn(input)
        output = self.act(output)
        return output


class C(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                              bias_attr=False, groups=groups)

    def forward(self, x):
        return self.conv(x)


class EESP(nn.Layer):
    """
    EESP block, principle:
        REDUCE->SPLIT->TRANSFORM->MERGE
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            stride=1,
            k=4,
            r_lim=7,
            down_method='esp'
            ):
        '''
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param stride: factor by which we should skip (useful for down-sampling). If 2, then down-samples the feature map by 2
        :param k: # of parallel branches
        :param r_lim: A maximum value of receptive field allowed for EESP block
        :param down_method: Downsample or not (equivalent to say stride is 2 or not)
        '''
        super(EESP, self).__init__()
        self.stride = stride
        n = int(out_channels / k)
        n1 = out_channels - (k - 1) * n
        assert down_method in ['avg', 'esp'], 'One of these is suppported (avg or esp)'
        assert n == n1, "n(={}) and n1(={}) should be equal for Depth-wise Convolution ".format(n, n1)
        self.proj_1x1 = ConvBNLayer(in_channels, n, 1, 1, k)

        # (For convenience) Mapping between dilation rate and receptive field for a 3x3 kernel
        map_receptive_ksize = {3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7, 17: 8}
        self.k_sizes = list()
        for i in range(k):
            ksize = int(3 + 2 * i)
            # After reaching the receptive field limit, fall back to the base kernel size of 3 with a dilation rate of 1
            ksize = ksize if ksize <= r_lim else 3
            self.k_sizes.append(ksize)

        self.k_sizes.sort()
        self.spp_dw = nn.LayerList()
        for i in range(k):
            d_rate = map_receptive_ksize[self.k_sizes[i]]
            self.spp_dw.append(CDilated(n, n, 3, stride, d=d_rate, groups=n))
        # Performing a group convolution with K groups is the same as performing K point-wise convolutions
        self.conv_1x1_exp = ConvBn(out_channels, out_channels, 1, 1, groups=k)
        self.br_after_cat = BnAct(out_channels)
        self.module_act = nn.PReLU()
        self.downAvg = True if down_method == 'avg' else False

        self.in_channels = in_channels

    def forward(self, x):
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]
        # compute the output for each branch and hierarchically fuse them
        # i.e. Split --> Transform --> HFF
        for k in range(1, len(self.spp_dw)):
            out_k = self.spp_dw[k](output1)
            # HFF
            out_k = out_k + output[k - 1]
            output.append(out_k)
        # Merge
        expanded = self.conv_1x1_exp(  # learn linear combinations using group point-wise convolutions
            self.br_after_cat(  # apply batch normalization followed by activation function (PRelu in this case)
                paddle.concat(output, 1)  # concatenate the output of different branches
            )
        )
        del output
        # if down-sampling, then return the concatenated vector
        # because Downsampling function will combine it with avg. pooled feature map and then threshold it
        if self.stride == 2 and self.downAvg:
            return expanded

        # if dimensions of input and concatenated vector are the same, add them (RESIDUAL LINK)
        if expanded.shape == x.shape:
            expanded = expanded + x

        # Threshold the feature map using activation function (PReLU in this case)
        x = self.module_act(expanded)
        return x


class PSPModule(nn.Layer):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 4, 8)):
        super().__init__()
        self.stages = []
        self.stages = nn.LayerList([C(features, features, 3, 1, groups=features) for size in sizes])
        self.project = ConvBNLayer(features * (len(sizes) + 1), out_features, 1, 1)

    def forward(self, feats):
        h, w = feats.shape[2], feats.shape[3]
        out = [feats]
        for stage in self.stages:
            feats = F.avg_pool2d(feats, kernel_size=3, stride=2, padding=1)
            upsampled = F.interpolate(stage(feats), size=(h, w), mode='bilinear', align_corners=True)
            out.append(upsampled)
        return self.project(paddle.concat(out, 1))


@manager.MODELS.add_component
class EESPNetSeg(nn.Layer):
    def __init__(self, num_classes=19, backbone=None, s=1):
        super().__init__()
        if s <= 0.5:
            p = 0.1
        else:
            p = 0.2
        self.backbone = backbone
        self.proj_L4_C = ConvBNLayer(256, 128, 1, 1)
        pspSize = 2 * 128
        self.pspMod = nn.Sequential(EESP(pspSize, pspSize // 2, stride=1, k=4, r_lim=7),
                                    PSPModule(pspSize // 2, pspSize // 2))
        self.esp1 = EESP(pspSize, pspSize // 2, stride=1, k=4, r_lim=7)
        self.psp1 = PSPModule(pspSize // 2, pspSize // 2)

        self.project_l3 = nn.Sequential(nn.Dropout2D(p=p), C(pspSize // 2, num_classes, 1, 1))
        self.act_l3 = BnAct(num_classes)
        self.project_l2 = ConvBNLayer(64 + num_classes, num_classes, 1, 1)
        self.project_l1 = nn.Sequential(nn.Dropout2D(p=p),
                                        C(32 + num_classes, num_classes, 1, 1))

    def hierarchicalUpsample(self, x, factor=3):
        for i in range(factor):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x

    def forward(self, x):
        logits_list = []
        out_l1, out_l2, out_l3, out_l4 = self.backbone(x)
        out_l4_proj = self.proj_L4_C(out_l4)
        up_l4_to_l3 = F.interpolate(out_l4_proj, scale_factor=2, mode='bilinear', align_corners=True)
        # merged_l3_upl4 = self.pspMod(paddle.concat([out_l3, up_l4_to_l3], 1))
        merged_l3_upl4 = self.esp1(paddle.concat([out_l3, up_l4_to_l3], 1))
        merged_l3_upl4 = self.psp1(merged_l3_upl4)

        proj_merge_l3_bef_act = self.project_l3(merged_l3_upl4)
        proj_merge_l3 = self.act_l3(proj_merge_l3_bef_act)
        out_up_l3 = F.interpolate(proj_merge_l3, scale_factor=2, mode='bilinear', align_corners=True)
        merge_l2 = self.project_l2(paddle.concat([out_l2, out_up_l3], 1))
        out_up_l2 = F.interpolate(merge_l2, scale_factor=2, mode='bilinear', align_corners=True)
        merge_l1 = self.project_l1(paddle.concat([out_l1, out_up_l2], 1))
        if self.training:
            logits_list.append(F.interpolate(merge_l1, scale_factor=2, mode='bilinear', align_corners=True)) 
            logits_list.append(self.hierarchicalUpsample(proj_merge_l3_bef_act))
        else:
            logits_list.append(F.interpolate(merge_l1, scale_factor=2, mode='bilinear', align_corners=True))
        return logits_list

if __name__ == '__main__':
    x = paddle.randn((4, 3, 512, 1024))
    backbone = EESPNet()
    y1 = backbone(x)
    for y_ in y1:
        print(x.shape, y_.shape)

    head = EESPNetSeg()
    y, t = head(y1)
    print(x.shape, y.shape, t.shape)