from torch import nn
import torch

class DefaultActivateEnv(object):
    creation_func = nn.ReLU
    creation_param = {}

def make_default_activate(override_param: dict = {}):
    param = DefaultActivateEnv.creation_param.copy()
    param.update(override_param)
    return DefaultActivateEnv.creation_func(**param)

def _process_activate(activate):
    if activate is not None and not isinstance(activate, (bool, nn.Module)):
        msg = 'activate can be None or False for no activation function\n' + \
              '    or can be True for default activation function\n' + \
              '    or can be an instance of nn.Module for specific activation\n' + \
              '    function provided'
        raise ValueError(msg)
    if activate is None or activate is False:
        return None
    elif activate is True:
        return make_default_activate()
    else:
        return activate

class BasicFC(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 bn=True,
                 activate=True,
                 **kwargs):
        super(BasicFC, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels, **kwargs)
        self.need_bn = bn
        self.act = _process_activate(activate)
        if self.need_bn:
            self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.fc(x)
        if self.need_bn:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class BasicConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 bn=True,
                 activate=True,
                 **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.need_bn = bn
        self.act = _process_activate(activate)
        if self.need_bn:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.need_bn:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

class OnnxReshape(nn.Module):
    def __init__(self, *args):
        super(OnnxReshape, self).__init__()
        count_minus = 0
        for dim in args:
            assert isinstance(dim, int) and dim >= -1
            if dim < 0:
                count_minus += 1
        assert count_minus <= 1

        self.shape = [dim for dim in args]
        self.shape_mask = [1 if dim == 0 else 0 for dim in args]
        # param_for_view = self.shape + self.shape_mask * input_shape[:self.rank]
        self.shape = torch.tensor(self.shape)
        self.shape_mask = torch.tensor(self.shape_mask)
        self.rank = len(self.shape)

    def forward(self, x):
        input_shape = torch.tensor(x.size())
        output_shape = self.shape + self.shape_mask * input_shape[:self.rank]
        return x.view(list(output_shape))

# SkipConnection(x) = op( F(x), proj/iden(x) )
# For Residual Block in Resnet18: op == '+', F == conv3x3 * 2
#     use 'iden' for common block, and 'proj' for down-sample block
#
# `F`: F(x) mentioned above
# `F_out_channels`: output channels of F(x), used to infer channels
#                   of post BatchNorm, can be None when `post_bn`
#                   == False
# `skip_connect_type`: available choices are 'add', 'mul', 'cat',
#                      which are addition, multiplication, concatenation
#                      correspondingly. Default is 'add'
# `use_proj`: whether to perform a linear projection by the skip connection,
#             Default is false, but must be true when dimensions of input
#             and output of `F` do not match
# `in_channels`: input channels of `F`, used when `use_proj` == True
# `out_channels`: output channels of projection, used when `use_proj` == True
# `stride`: downsample rate of `F`, used when `use_proj` == True
# `post_bn`: apply BatchNorm after `op`, default is False
# `post_activate`: apply an activation function at last (provided as
#                  a nn.Module, or True for default activation function),
#                  default is True
class SkipConnection(nn.Module):
    def __init__(self, F: nn.Module, F_out_channles: int = None,
                 skip_connect_type: str = 'add', use_proj: bool = False,
                 in_channels: int = None, out_channels: int = None, stride: int = None,
                 post_bn: bool = False, post_activate=True):
        super(SkipConnection, self).__init__()

        # skip connnection type
        if skip_connect_type not in ('add', 'mul', 'cat'):
            raise ValueError("skip_connect_type must be one of 'add', 'mul', 'cat'")

        self.skip_type_ = skip_connect_type
        if self.skip_type_ == 'add':
            self.skip_op = SkipConnection.__skip_add
        elif self.skip_type_ == 'mul':
            self.skip_op = SkipConnection.__skip_mul
        else:
            self.skip_op = SkipConnection.__skip_cat

        def expect_postive_int(val, name: str):
            if not isinstance(val, int) or val <= 0:
                raise ValueError('%s should be positive integer' % name)

        # for projection on the skip connection
        self.use_proj_ = use_proj
        if self.use_proj_:
            expect_postive_int(in_channels, 'in_channels')
            expect_postive_int(out_channels, 'out_channels')
            expect_postive_int(stride, 'stride')
            self.proj = BasicConv2d(in_channels, out_channels, bn=True, activate=None, bias=False, kernel_size=1, padding=0, stride=stride)
        else:
            self.proj = None

        # post bn
        self.use_post_bn_ = post_bn
        if self.use_post_bn_:
            expect_postive_int(F_out_channles, 'F_out_channles')
            if self.skip_type_ != 'cat':
                assert self.proj is None or out_channels == F_out_channles
                final_channels = F_out_channles
            else:
                final_channels = F_out_channles + out_channels if self.use_proj_ else 2 * F_out_channles

        self.post_bn = None if not self.use_post_bn_ else \
            nn.BatchNorm2d(final_channels)

        # other
        self.post_activate = _process_activate(post_activate)
        self.F = F

    def forward(self, x):
        y = self.F(x)
        if self.use_proj_:
            x = self.proj(x)

        x = self.skip_op(x, y)

        if self.use_post_bn_:
            x = self.post_bn(x)
        if self.post_activate is not None:
            x = self.post_activate(x)

        return x

    def __skip_add(x, y):
        return x + y

    def __skip_mul(x, y):
        return x * y

    def __skip_cat(x, y):
        return torch.cat((x, y), 1)


# ResNetBlock_3_3 (used in ResNet-18, ResNet-34):
# SkipConnection(x) = op( F(x), proj/iden(x) ), where op == 'add',
#     F is 2 stacked conv3x3, proj is used when downsample
#
# `in_channels`: input channels of this block
# `out_channels`: output channels of this block
# `stride`: downsample rate of this block, default is 1, e.g. no
#           downsample
# `post_bn`: apply BatchNorm after `op`, default is False
# `post_activate`: apply an activation function at last, default is True
class ResNetBlock_3_3(SkipConnection):

    class Double_Conv3x3(nn.Module):
        def __init__(self, stride, in_channels, out_channels):
            super(ResNetBlock_3_3.Double_Conv3x3, self).__init__()

            self.conv1 = BasicConv2d(
                in_channels, out_channels, bn=True, activate=True,
                bias=False, kernel_size=3, padding=1, stride=stride
            )

            self.conv2 = BasicConv2d(
                out_channels, out_channels, bn=True, activate=None,
                bias=False, kernel_size=3, padding=1, stride=1
            )

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            return x

    def __init__(self,
                 in_channels: int, out_channels: int, stride: int = 1,
                 post_bn: bool = False, post_activate: bool = True):
        use_proj = in_channels != out_channels or stride != 1

        super(ResNetBlock_3_3, self).__init__(
            F=ResNetBlock_3_3.Double_Conv3x3(
                stride, in_channels, out_channels
            ),
            F_out_channles=out_channels,
            skip_connect_type='add', use_proj=use_proj,
            in_channels=in_channels, out_channels=out_channels, stride=stride,
            post_bn=post_bn,
            post_activate=post_activate
        )


# ResNetBlock_1_3_1 (used in ResNet-50, ResNet-101, ResNet-152):
# SkipConnection(x) = op( F(x), proj/iden(x) ), where op == 'add',
#     F is conv1x1-conv3x3-conv1x1, proj is used when downsample
#
# `in_channels`: input channels of this block
# `out_channels`: output channels of this block
# `stride`: downsample rate of this block, default is 1, e.g. no
#           downsample
# `post_bn`: apply BatchNorm after `op`, default is False
# `post_activate`: apply an activation function at last, default is True
# `squeeze_ratio`: default is 4, ratio defined by
#                  (out_channels of F)/(out_channels of the inner conv3x3)
# `min_inner_channels`: least in_channels & out_channels the inner conv3x3
#                       should have, default is 8
class ResNetBlock_1_3_1(SkipConnection):

    class Stacked_Conv_1_3_1(nn.Module):
        def __init__(self, stride, in_channels, out_channels, squeeze_ratio, min_inner_channels):
            inner_channels = max(
                min_inner_channels,
                int(out_channels / squeeze_ratio)
            )
            super(ResNetBlock_1_3_1.Stacked_Conv_1_3_1, self).__init__()

            self.squeeze = BasicConv2d(
                in_channels, inner_channels, bn=True, activate=True,
                bias=False, kernel_size=1, padding=0, stride=stride
            )
            self.conv = BasicConv2d(
                inner_channels, inner_channels, bn=True, activate=True,
                bias=False, kernel_size=3, padding=1, stride=1
            )
            self.linear = BasicConv2d(
                inner_channels, out_channels, bn=True, activate=None,
                bias=False, kernel_size=1, padding=0, stride=1
            )

        def forward(self, x):
            x = self.squeeze(x)
            x = self.conv(x)
            x = self.linear(x)
            return x

    def __init__(self,
                 in_channels: int, out_channels: int, stride: int = 1,
                 post_bn: bool = False, post_activate: bool = True,
                 squeeze_ratio: float = 4, min_inner_channels: int = 8):
        use_proj = in_channels != out_channels or stride != 1

        super(ResNetBlock_1_3_1, self).__init__(
            F=ResNetBlock_1_3_1.Stacked_Conv_1_3_1(
                stride, in_channels, out_channels,
                squeeze_ratio, min_inner_channels
            ),
            F_out_channles=out_channels,
            skip_connect_type='add', use_proj=use_proj,
            in_channels=in_channels, out_channels=out_channels, stride=stride,
            post_bn=post_bn,
            post_activate=post_activate
        )

class ResNetBase(nn.Module):
    def __init__(self):
        super(ResNetBase, self).__init__()

    def make_stacked_resblock33(self, block_count: int,
                                in_channels: int, out_channels: int,
                                downsample: bool = False):
        mod_list = [ResNetBlock_3_3(
            in_channels, out_channels, 2 if downsample else 1,
        )]
        for _ in range(block_count - 1):
            mod_list.append(ResNetBlock_3_3(
                out_channels, out_channels, 1,
            ))
        return nn.Sequential(*mod_list)

    def make_stacked_resblock131(self, block_count: int,
                                 in_channels: int, out_channels: int,
                                 downsample: bool = False,
                                 squeeze_ratio: float = 4):
        mod_list = [ResNetBlock_1_3_1(
            in_channels, out_channels, 2 if downsample else 1,
            squeeze_ratio=squeeze_ratio
        )]
        for _ in range(block_count - 1):
            mod_list.append(ResNetBlock_1_3_1(
                out_channels, out_channels, 1,
                squeeze_ratio=squeeze_ratio
            ))
        return nn.Sequential(*mod_list)

def _get_kernel_stride_pad(isize: int, osize: int):
    # assume isize >= osize
    if isize == osize:
        return 1, 1, 0
    elif isize < osize:
        raise ValueError('isize must not be smaller than osize')

    div, rem = divmod(isize, osize)
    # try pad up to isize after pad being divisible by osize
    # reach to divisible, then there is no overlap, further
    #     padding will not gain any benifit, as overlap is not
    #     zero and pad also become larger
    pad_upper = osize - rem
    del div, rem

    min_target = 2 << 31
    argmin_pad = None

    for pad in range(pad_upper + 1):  # inclusive
        isize_pad = isize + pad
        div, rem  = divmod(isize_pad, osize)

        # ksize = div + rem
        # ksize must be strictly larger than pad of any side,
        #     i.e. ksize > right_pad
        # as   right_pad + left_pad == pad,
        # and  right_pad == left_pad when pad is even
        # and  right_pad == left_pad + 1 when pad is odd
        # otherwise, the last output will be fully located in
        #     padding region
        if div + rem <= pad // 2 + pad % 2:
            continue

        # overlap between to output is
        #     ksize - stride = div + rem - div = rem
        # there are (osize - 1) of them
        overlap   = rem * (osize - 1)
        target    = overlap + pad
        # print('pad=%d overlap=%d target=%d' % (
        #     pad, overlap, target
        # ))

        if target < min_target:  # pad must larger than previous
            min_target = target
            argmin_pad = pad

    assert argmin_pad is not None

    div, rem = divmod(isize + argmin_pad, osize)

    ksize    = div + rem
    stride   = div
    pad      = argmin_pad // 2  # ceil_mode = True

    return ksize, stride, pad

class ResNet12_Backbone(ResNetBase):

    RECOMM_INPUT = 64

    def __init__(self, gray: bool = False):
        super(ResNet12_Backbone, self).__init__()

        self.gray  = bool(gray)
        channels   = [48, 24, 48, 96, 80]
        blocks     = [0, 1, 2, 2, 0]
        self.conv1 = BasicConv2d(1 if gray else 3, channels[0],
                                 bn=True, activate=True, bias=False,
                                 kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)

        self.conv2 = self.make_stacked_resblock33(blocks[1],
                                                  channels[0], channels[1])
        self.conv3 = self.make_stacked_resblock33(blocks[2],
                                                  channels[1], channels[2],
                                                  downsample=True)
        self.conv4 = self.make_stacked_resblock33(blocks[3],
                                                  channels[2], channels[3],
                                                  downsample=True)
        self.conv5 = BasicConv2d(channels[3], channels[4],
                                 bn=True, activate=True, bias=False,
                                 kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class TaskBranch(nn.Sequential):

    def __init__(self, branch_hidden: int, hidden_dim: int, task_dim: int,
                 input_dim: int = None):

        if input_dim is None:
            input_dim = hidden_dim

        assert branch_hidden >= 0
        assert hidden_dim > 0 and task_dim > 0
        assert input_dim > 0

        args   = {'bn': True, 'bias': False, 'activate': True}

        front  = [] if branch_hidden == 0 \
            else [BasicFC(input_dim, hidden_dim, **args)]
        body   = [BasicFC(hidden_dim, hidden_dim, **args)
                  for _ in range(branch_hidden - 1)]
        back   = [BasicFC(hidden_dim, task_dim, bn=False, bias=True, activate=None)]

        layers = front + body + back
        super(TaskBranch, self).__init__(*layers)

class ResNet12(nn.Module):
    def __init__(self, num_classes=1000, dropout_factor=0.2, img_size=224, pool_size=2, hidden_num=2, hidden_dim=128):
        super(ResNet12, self).__init__()
        self.backbone = ResNet12_Backbone()

        c, h, w = self.get_out_size(self.backbone, img_size)
        #print(c, h, w, pool_size)

        kh, sh, ph = _get_kernel_stride_pad(h, pool_size)
        self.avg_pool = nn.AvgPool2d(kernel_size=(kh, kh), stride=(sh, sh), padding=(ph, ph),
                                     ceil_mode=True,
                                     count_include_pad=False)

        self.flatten = OnnxReshape(0, -1)

        pool_numel = c * pool_size * pool_size
        self.shared_hidden = BasicFC(pool_numel, hidden_dim, bn=True, bias=False, activate=True)
        branch_hidden = hidden_num - 1 if hidden_num > 1 else 0
        branch_input = pool_numel if hidden_num == 0 else hidden_dim

        self.pts = TaskBranch(branch_hidden, hidden_dim, num_classes, branch_input)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.shared_hidden(x)
        pts = self.pts(x)
        return pts

    def get_out_size(self, backbone, img_size):
        backbone.to('cpu')

        x = torch.rand(1, 3, img_size, img_size, dtype=torch.float32, device='cpu')
        y = backbone(x)
        #print("111111111", y.shape)

        return y.size(1), y.size(2), y.size(3)
