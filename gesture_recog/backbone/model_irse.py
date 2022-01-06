import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, \
    AdaptiveAvgPool2d, Sequential, Module
from collections import namedtuple


# Support: ['IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']

class GNAP(Module):
    def __init__(self, embedding_size):
        super(GNAP, self).__init__()
        assert embedding_size == 512
        self.bn1 = BatchNorm2d(512, affine=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn2 = BatchNorm1d(512, affine=False)

    def forward(self, x):
        x = self.bn1(x)

        x_norm = torch.norm(x, 2, 1, True)
        x_norm_mean = torch.mean(x_norm)
        weight = x_norm_mean / x_norm
        x = x * weight

        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        feature = self.bn2(x)
        return feature

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight.data)

        #self.relu = ReLU(inplace=True)
        self.relu = ReLU(inplace=False)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)

        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), 
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), 
            BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        output = res + shortcut
        # return res + shortcut
        return output


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        output = res + shortcut
        # return res + shortcut
        return output


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):

    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]

    return blocks


class Backbone(Module):
    def __init__(self, input_size=[224,224], num_layers=50, mode='ir', output_name="GDC", return_features=False):
        super(Backbone, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        self.return_features = return_features
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        modules = []
        for bottleneck in blocks[0]:
            modules.append(
                unit_module(bottleneck.in_channel,
                            bottleneck.depth,
                            bottleneck.stride))
        self.layer0 = Sequential(*modules)

        modules = []
        for bottleneck in blocks[1]:
            modules.append(
                unit_module(bottleneck.in_channel,
                            bottleneck.depth,
                            bottleneck.stride))
        self.layer1 = Sequential(*modules)

        modules = []
        for bottleneck in blocks[2]:
            modules.append(
                unit_module(bottleneck.in_channel,
                            bottleneck.depth,
                            bottleneck.stride))
        self.layer2 = Sequential(*modules)

        modules = []
        for bottleneck in blocks[3]:
            modules.append(
                unit_module(bottleneck.in_channel,
                            bottleneck.depth,
                            bottleneck.stride))
        self.layer3 = Sequential(*modules)

        if output_name == "GDC":
            if input_size[0] == 112:
                self.output_layer = Sequential(BatchNorm2d(512),
                                               Dropout(0.4),
                                               Flatten(),
                                               Linear(512 * 7 * 7, 512),
                                               BatchNorm1d(512, affine=False))
            else:
                self.output_layer = Sequential(BatchNorm2d(512),
                                               Dropout(0.4),
                                               Flatten(),
                                               Linear(512 * 14 * 14, 512),
                                               BatchNorm1d(512, affine=False))
        elif output_name == "GNAP":
            self.output_layer = GNAP(512)
        else:
            raise ValueError(output_name)

        self._initialize_weights()

    def forward(self, x):
        feature_list = []

        x = self.input_layer(x)
        feature_list.append(x)

        x = self.layer0(x)
        feature_list.append(x)

        x = self.layer1(x)
        feature_list.append(x)

        x = self.layer2(x)
        feature_list.append(x)

        x = self.layer3(x)
        feature_list.append(x)

        x = self.output_layer(x)

        if self.return_features:
            return x, feature_list
        else:
            return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

def IR_50(input_size=[224,224], **kwargs):
    """Constructs a ir-50 model.
    """
    model = Backbone(input_size, 50, 'ir', **kwargs)

    return model


def IR_101(input_size=[224,224]):
    """Constructs a ir-101 model.
    """
    model = Backbone(input_size, 100, 'ir')

    return model


def IR_152(input_size=[224,224]):
    """Constructs a ir-152 model.
    """
    model = Backbone(input_size, 152, 'ir')

    return model


def IR_SE_50(input_size=[224,224]):
    """Constructs a ir_se-50 model.i
    """
    model = Backbone(input_size, 50, 'ir_se')

    return model


def IR_SE_101(input_size=[224,224]):
    """Constructs a ir_se-101 model.
    """
    model = Backbone(input_size, 100, 'ir_se')

    return model


def IR_SE_152(input_size=[224,224]):
    """Constructs a ir_se-152 model.
    """
    model = Backbone(input_size, 152, 'ir_se')

    return model

