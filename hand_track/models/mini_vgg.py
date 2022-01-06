from torch import nn
from collections import OrderedDict
from typing import List, Tuple
import torch

class DefaultActivateEnv(object):
    creation_func = nn.ReLU
    creation_param = {}

# Context manager to modify the default activation creation function
# together with parameters called with. Use it with 'with' statement
#
# `creation_func`: callable, the function to create the activation
#                  instance, usually a type (nn.Sigmoid, for example),
#                  which is treat as constructor
# `creation_param`: dict, 'kwargs' storing the parameters that will be
#                   passed to `creation_func` each time making new
#                   instance of activation function
class DefaultActivate(object):
    def __init__(self, creation_func=nn.ReLU, creation_param: dict = {}):
        if not callable(creation_func):
            raise ValueError('creation_func should be callable')
        if not isinstance(creation_param, dict):
            raise ValueError('creation_param should be dict')

        tmp = creation_func(**creation_param)
        if not isinstance(tmp, nn.Module):
            raise ValueError('the activation function should be implemented in nn.Module')

        self.creation_func = creation_func
        self.creation_param = creation_param

    def __enter__(self):
        self.func_bak = DefaultActivateEnv.creation_func
        self.param_bak = DefaultActivateEnv.creation_param
        DefaultActivateEnv.creation_func = self.creation_func
        DefaultActivateEnv.creation_param = self.creation_param

    def __exit__(self, exc_type, exc_val, exc_tb):
        DefaultActivateEnv.creation_func = self.func_bak
        DefaultActivateEnv.creation_param = self.param_bak

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

class MiniVGG_Backbone(nn.Sequential):

    RECOMM_INPUT = 112

    def __init__(self, gray: bool = False, prelu: bool = True):

        self.__prelu_ch = 1

        with DefaultActivate(self.__get_prelu):

            layers          = []
            self.__prelu_ch = 8
            layers.append(('conv_pre', BasicConv2d(
                1 if gray else 3, 8,
                bn=True, activate=True, bias=False,
                kernel_size=5, stride=2, padding=0
            )))
            layers += self.__make_stage('stage1', 8, 16, 16, is_pad=False)
            layers += self.__make_stage('stage2', 16, 24, 24, is_pad=False)
            layers += self.__make_stage('stage3', 24, 40, 80, is_pad=True)

            super().__init__(OrderedDict(layers))

    def __get_prelu(self) -> nn.PReLU:
        ret = nn.PReLU(self.__prelu_ch)
        self.__prelu_ch = None
        return ret

    def __make_stage(self, prefix: str, in_channels: int,
                     mid_channels: int, out_channels: int,
                     ksize: int = 3, is_pad: bool = True
                     ) -> List[Tuple[str, nn.Module]]:

        ret     = []
        padding = ksize // 2 if is_pad else 0

        ret.append((prefix + '_pool', nn.MaxPool2d(
            2, 2, 0, ceil_mode=True
        )))

        self.__prelu_ch = mid_channels

        ret.append((prefix + '_conv1', BasicConv2d(
            in_channels, mid_channels,
            bn=True, activate=True, bias=False,
            kernel_size=ksize, padding=padding, stride=1
        )))

        self.__prelu_ch = out_channels

        ret.append((prefix + '_conv2', BasicConv2d(
            mid_channels, out_channels,
            bn=True, activate=True, bias=False,
            kernel_size=ksize, padding=padding, stride=1
        )))

        return ret

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

class MiniVGG(nn.Module):
    def __init__(self, num_classes=1000, dropout_factor=0.2, img_size=224, pool_size=2, hidden_num=2, hidden_dim=128):
        super(MiniVGG, self).__init__()
        self.backbone = MiniVGG_Backbone()

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

        return y.size(1), y.size(2), y.size(3)
