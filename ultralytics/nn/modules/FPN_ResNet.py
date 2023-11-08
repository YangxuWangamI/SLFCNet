import paddle
from paddle import nn
from paddle.utils.download import get_weights_path_from_url

__all__ = []

model_urls = {
    'resnet50': (
        'https://paddle-hapi.bj.bcebos.com/models/resnet50.pdparams',
        'ca6f485ee1ab0492d38f323885b0ad80',
    )
}


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            groups=1,
            base_width=64,
            dilation=1,
            norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D

        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock"
            )

        self.conv1 = nn.Conv2D(
            inplanes, planes, 3, padding=1, stride=stride, bias_attr=False
        )
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(planes, planes, 3, padding=1, bias_attr=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Layer):
    expansion = 4

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            groups=1,
            base_width=64,
            dilation=1,
            norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = nn.Conv2D(inplanes, width, 1, bias_attr=False)
        self.bn1 = norm_layer(width)

        self.conv2 = nn.Conv2D(
            width,
            width,
            3,
            padding=dilation,
            stride=stride,
            groups=groups,
            dilation=dilation,
            bias_attr=False,
        )
        self.bn2 = norm_layer(width)

        self.conv3 = nn.Conv2D(
            width, planes * self.expansion, 1, bias_attr=False
        )
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Layer):
    """ResNet model from
    Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>

    """

    def __init__(
            self,
            block,
            depth=50,
            width=64,
            num_classes=2,
            with_pool=True,
            groups=1,
    ):
        super().__init__()
        layer_cfg = {
            50: [3, 4, 6, 3]
        }
        layers = layer_cfg[depth]
        self.groups = groups
        self.base_width = width
        self.num_classes = num_classes
        self.with_pool = with_pool
        self._norm_layer = nn.BatchNorm2D

        self.inplanes = 64
        self.dilation = 1

        self.conv1 = nn.Conv2D(
            3,
            self.inplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False,
        )
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if with_pool:
            self.avgpool = nn.AdaptiveAvgPool2D((1, 1))

        if num_classes > 0:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(
                    self.inplanes,
                    planes * block.expansion,
                    1,
                    stride=stride,
                    bias_attr=False,
                ),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = x4

        if self.with_pool:
            x = self.avgpool(x)

        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            out = self.fc(x)

        return [x1, x2, x3, x4, out]


def _resnet(arch, num_classes, Block, depth, pretrained, **kwargs):
    model = ResNet(Block, depth, num_classes=num_classes, **kwargs)
    if pretrained:
        assert (
                arch in model_urls
        ), "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch
        )
        weight_path = get_weights_path_from_url(
            model_urls[arch][0], model_urls[arch][1]
        )

        param = paddle.load(weight_path)
        model.set_dict(param)

    return model


def resnet50(num_classes, pretrained=False, **kwargs):
    return _resnet('resnet50', num_classes, BottleneckBlock, 50, pretrained, **kwargs)


# 构建Conv+BN+ReLU块：
class ConvBnReLU(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding, bias_attr=False)
        self.bn = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# 构建FPN网络
class FPN(nn.Layer):
    def __init__(self, backbone_out_channels=[256, 512, 1024, 2048]):
        super(FPN, self).__init__()
        self.backbone_out_channels = backbone_out_channels

        # 定义 1x1的卷积操作，用于调整通道数 变为256
        self.conv1 = ConvBnReLU(self.backbone_out_channels[0], 256)
        self.conv2 = ConvBnReLU(self.backbone_out_channels[1], 256)
        self.conv3 = ConvBnReLU(self.backbone_out_channels[2], 256)
        self.conv4 = ConvBnReLU(self.backbone_out_channels[3], 256)

        # 定义 FPN 中的上采样和下采样操作
        self.upsample = nn.Upsample(scale_factor=2.0, mode='nearest')

        # 定义 3x3的卷积操作 平滑特征图
        self.smooth = nn.Conv2D(256, 256, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        input1, input2, input3, input4, _ = inputs

        # 定义不同层级的特征图
        # input1 R1  256 -> 256
        # input2 R2  512 -> 256
        # input3 R3  1024  -> 256
        # input4 R4  2048  -> 256
        r1 = self.conv1(input1)
        r2 = self.conv2(input2)
        r3 = self.conv3(input3)
        r4 = self.conv4(input4)

        # 定义不同层级的特征图融合
        f4 = r4
        f3 = r3 + self.upsample(r4)  # 32 -> 64 + 64 -> 64
        f2 = r2 + self.upsample(f3)  # 64 -> 128 + 128 -> 128
        f1 = r1 + self.upsample(f2)  # 128 -> 256 + 256 -> 256

        return self.smooth(f1), self.smooth(f2), self.smooth(f3), self.smooth(f4)

if __name__ == '__main__':

    resnet50_model = resnet50(2, pretrained=False)
    x = paddle.rand([2, 3, 256, 256])
    out = resnet50_model(x)
    fpn = FPN()
    y = fpn(out)
    for i in y:
        print(i.shape)
