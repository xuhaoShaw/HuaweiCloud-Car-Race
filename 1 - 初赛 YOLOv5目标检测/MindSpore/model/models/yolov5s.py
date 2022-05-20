import numpy as np
import mindspore
from mindspore import nn
from mindspore import Tensor, Parameter, context
from mindspore import dtype as ms
from models.loss import ConfidenceLoss, ClassLoss
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size

class Module1(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_0_kernel_size, conv2d_0_stride,
                 conv2d_0_padding, conv2d_0_pad_mode):
        super(Module1, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=conv2d_0_kernel_size,
                                  stride=conv2d_0_stride,
                                  padding=conv2d_0_padding,
                                  pad_mode=conv2d_0_pad_mode,
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.sigmoid_1 = nn.Sigmoid()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(P.Cast()(x, ms.float32))
        opt_sigmoid_1 = self.sigmoid_1(opt_conv2d_0)
        opt_mul_2 = P.Mul()(opt_conv2d_0, opt_sigmoid_1)
        return opt_mul_2


class Module12(nn.Cell):
    def __init__(self, module1_0_conv2d_0_in_channels, module1_0_conv2d_0_out_channels, module1_0_conv2d_0_kernel_size,
                 module1_0_conv2d_0_stride, module1_0_conv2d_0_padding, module1_0_conv2d_0_pad_mode,
                 module1_1_conv2d_0_in_channels, module1_1_conv2d_0_out_channels, module1_1_conv2d_0_kernel_size,
                 module1_1_conv2d_0_stride, module1_1_conv2d_0_padding, module1_1_conv2d_0_pad_mode):
        super(Module12, self).__init__()
        self.module1_0 = Module1(conv2d_0_in_channels=module1_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module1_0_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module1_0_conv2d_0_kernel_size,
                                 conv2d_0_stride=module1_0_conv2d_0_stride,
                                 conv2d_0_padding=module1_0_conv2d_0_padding,
                                 conv2d_0_pad_mode=module1_0_conv2d_0_pad_mode)
        self.module1_1 = Module1(conv2d_0_in_channels=module1_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module1_1_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module1_1_conv2d_0_kernel_size,
                                 conv2d_0_stride=module1_1_conv2d_0_stride,
                                 conv2d_0_padding=module1_1_conv2d_0_padding,
                                 conv2d_0_pad_mode=module1_1_conv2d_0_pad_mode)

    def construct(self, x):
        module1_0_opt = self.module1_0(x)
        module1_1_opt = self.module1_1(module1_0_opt)
        return module1_1_opt


class Module11(nn.Cell):
    def __init__(self, module1_0_conv2d_0_in_channels, module1_0_conv2d_0_out_channels, module1_0_conv2d_0_kernel_size,
                 module1_0_conv2d_0_stride, module1_0_conv2d_0_padding, module1_0_conv2d_0_pad_mode,
                 module1_1_conv2d_0_in_channels, module1_1_conv2d_0_out_channels, module1_1_conv2d_0_kernel_size,
                 module1_1_conv2d_0_stride, module1_1_conv2d_0_padding, module1_1_conv2d_0_pad_mode,
                 module1_2_conv2d_0_in_channels, module1_2_conv2d_0_out_channels, module1_2_conv2d_0_kernel_size,
                 module1_2_conv2d_0_stride, module1_2_conv2d_0_padding, module1_2_conv2d_0_pad_mode):
        super(Module11, self).__init__()
        self.module1_0 = Module1(conv2d_0_in_channels=module1_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module1_0_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module1_0_conv2d_0_kernel_size,
                                 conv2d_0_stride=module1_0_conv2d_0_stride,
                                 conv2d_0_padding=module1_0_conv2d_0_padding,
                                 conv2d_0_pad_mode=module1_0_conv2d_0_pad_mode)
        self.module1_1 = Module1(conv2d_0_in_channels=module1_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module1_1_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module1_1_conv2d_0_kernel_size,
                                 conv2d_0_stride=module1_1_conv2d_0_stride,
                                 conv2d_0_padding=module1_1_conv2d_0_padding,
                                 conv2d_0_pad_mode=module1_1_conv2d_0_pad_mode)
        self.module1_2 = Module1(conv2d_0_in_channels=module1_2_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module1_2_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module1_2_conv2d_0_kernel_size,
                                 conv2d_0_stride=module1_2_conv2d_0_stride,
                                 conv2d_0_padding=module1_2_conv2d_0_padding,
                                 conv2d_0_pad_mode=module1_2_conv2d_0_pad_mode)

    def construct(self, x):
        module1_0_opt = self.module1_0(x)
        module1_1_opt = self.module1_1(module1_0_opt)
        module1_2_opt = self.module1_2(module1_1_opt)
        opt_add_0 = P.Add()(module1_0_opt, module1_2_opt)
        return opt_add_0


class Module15(nn.Cell):
    def __init__(self, module1_0_conv2d_0_in_channels, module1_0_conv2d_0_out_channels, module1_0_conv2d_0_kernel_size,
                 module1_0_conv2d_0_stride, module1_0_conv2d_0_padding, module1_0_conv2d_0_pad_mode,
                 module1_1_conv2d_0_in_channels, module1_1_conv2d_0_out_channels, module1_1_conv2d_0_kernel_size,
                 module1_1_conv2d_0_stride, module1_1_conv2d_0_padding, module1_1_conv2d_0_pad_mode,
                 module1_2_conv2d_0_in_channels, module1_2_conv2d_0_out_channels, module1_2_conv2d_0_kernel_size,
                 module1_2_conv2d_0_stride, module1_2_conv2d_0_padding, module1_2_conv2d_0_pad_mode):
        super(Module15, self).__init__()
        self.module1_0 = Module1(conv2d_0_in_channels=module1_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module1_0_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module1_0_conv2d_0_kernel_size,
                                 conv2d_0_stride=module1_0_conv2d_0_stride,
                                 conv2d_0_padding=module1_0_conv2d_0_padding,
                                 conv2d_0_pad_mode=module1_0_conv2d_0_pad_mode)
        self.module1_1 = Module1(conv2d_0_in_channels=module1_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module1_1_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module1_1_conv2d_0_kernel_size,
                                 conv2d_0_stride=module1_1_conv2d_0_stride,
                                 conv2d_0_padding=module1_1_conv2d_0_padding,
                                 conv2d_0_pad_mode=module1_1_conv2d_0_pad_mode)
        self.module1_2 = Module1(conv2d_0_in_channels=module1_2_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module1_2_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module1_2_conv2d_0_kernel_size,
                                 conv2d_0_stride=module1_2_conv2d_0_stride,
                                 conv2d_0_padding=module1_2_conv2d_0_padding,
                                 conv2d_0_pad_mode=module1_2_conv2d_0_pad_mode)

    def construct(self, x):
        module1_0_opt = self.module1_0(x)
        module1_1_opt = self.module1_1(module1_0_opt)
        module1_2_opt = self.module1_2(module1_1_opt)
        return module1_2_opt


class Module8(nn.Cell):
    def __init__(self, stridedslice_0_begin, stridedslice_0_end):
        super(Module8, self).__init__()
        self.stridedslice_0 = P.StridedSlice()
        self.stridedslice_0_begin = stridedslice_0_begin
        self.stridedslice_0_end = stridedslice_0_end
        self.stridedslice_0_strides = (1, 1, 1, 1, 1)
        self.mul_1_w = 2.0

    def construct(self, x):
        opt_stridedslice_0 = self.stridedslice_0(x, self.stridedslice_0_begin, self.stridedslice_0_end,
                                                 self.stridedslice_0_strides)
        opt_mul_1 = opt_stridedslice_0 * self.mul_1_w
        return opt_mul_1


class Model(nn.Cell):
    def __init__(self, img_size=1024, bs=1):  # bs - batch_size
        super(Model, self).__init__()

        self.stridedslice_0 = P.StridedSlice()
        self.stridedslice_0_begin = (0, 0, 0, 0)
        self.stridedslice_0_end = (bs, 3, img_size, img_size)
        self.stridedslice_0_strides = (1, 1, 2, 1)
        self.stridedslice_4 = P.StridedSlice()
        self.stridedslice_4_begin = (0, 0, 0, 0)
        self.stridedslice_4_end = (bs, 3, img_size//2, img_size)
        self.stridedslice_4_strides = (1, 1, 1, 2)
        self.stridedslice_1 = P.StridedSlice()
        self.stridedslice_1_begin = (0, 0, 1, 0)
        self.stridedslice_1_end = (bs, 3, img_size, img_size)
        self.stridedslice_1_strides = (1, 1, 2, 1)
        self.stridedslice_5 = P.StridedSlice()
        self.stridedslice_5_begin = (0, 0, 0, 0)
        self.stridedslice_5_end = (bs, 3, img_size//2, img_size)
        self.stridedslice_5_strides = (1, 1, 1, 2)
        self.stridedslice_2 = P.StridedSlice()
        self.stridedslice_2_begin = (0, 0, 0, 0)
        self.stridedslice_2_end = (bs, 3, img_size, img_size)
        self.stridedslice_2_strides = (1, 1, 2, 1)
        self.stridedslice_6 = P.StridedSlice()
        self.stridedslice_6_begin = (0, 0, 0, 1)
        self.stridedslice_6_end = (bs, 3, img_size//2, img_size)
        self.stridedslice_6_strides = (1, 1, 1, 2)
        self.stridedslice_3 = P.StridedSlice()
        self.stridedslice_3_begin = (0, 0, 1, 0)
        self.stridedslice_3_end = (bs, 3, img_size, img_size)
        self.stridedslice_3_strides = (1, 1, 2, 1)
        self.stridedslice_7 = P.StridedSlice()
        self.stridedslice_7_begin = (0, 0, 0, 1)
        self.stridedslice_7_end = (bs, 3, img_size//2, img_size)
        self.stridedslice_7_strides = (1, 1, 1, 2)
        self.concat_8 = P.Concat(axis=1)
        self.module12_0 = Module12(module1_0_conv2d_0_in_channels=12,
                                   module1_0_conv2d_0_out_channels=32,
                                   module1_0_conv2d_0_kernel_size=(3, 3),
                                   module1_0_conv2d_0_stride=(1, 1),
                                   module1_0_conv2d_0_padding=(1, 1, 1, 1),
                                   module1_0_conv2d_0_pad_mode="pad",
                                   module1_1_conv2d_0_in_channels=32,
                                   module1_1_conv2d_0_out_channels=64,
                                   module1_1_conv2d_0_kernel_size=(3, 3),
                                   module1_1_conv2d_0_stride=(2, 2),
                                   module1_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module1_1_conv2d_0_pad_mode="pad")
        self.module11_0 = Module11(module1_0_conv2d_0_in_channels=64,
                                   module1_0_conv2d_0_out_channels=32,
                                   module1_0_conv2d_0_kernel_size=(1, 1),
                                   module1_0_conv2d_0_stride=(1, 1),
                                   module1_0_conv2d_0_padding=0,
                                   module1_0_conv2d_0_pad_mode="valid",
                                   module1_1_conv2d_0_in_channels=32,
                                   module1_1_conv2d_0_out_channels=32,
                                   module1_1_conv2d_0_kernel_size=(1, 1),
                                   module1_1_conv2d_0_stride=(1, 1),
                                   module1_1_conv2d_0_padding=0,
                                   module1_1_conv2d_0_pad_mode="valid",
                                   module1_2_conv2d_0_in_channels=32,
                                   module1_2_conv2d_0_out_channels=32,
                                   module1_2_conv2d_0_kernel_size=(3, 3),
                                   module1_2_conv2d_0_stride=(1, 1),
                                   module1_2_conv2d_0_padding=(1, 1, 1, 1),
                                   module1_2_conv2d_0_pad_mode="pad")
        self.module1_0 = Module1(conv2d_0_in_channels=64,
                                 conv2d_0_out_channels=32,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=0,
                                 conv2d_0_pad_mode="valid")
        self.concat_28 = P.Concat(axis=1)
        self.module12_1 = Module12(module1_0_conv2d_0_in_channels=64,
                                   module1_0_conv2d_0_out_channels=64,
                                   module1_0_conv2d_0_kernel_size=(1, 1),
                                   module1_0_conv2d_0_stride=(1, 1),
                                   module1_0_conv2d_0_padding=0,
                                   module1_0_conv2d_0_pad_mode="valid",
                                   module1_1_conv2d_0_in_channels=64,
                                   module1_1_conv2d_0_out_channels=128,
                                   module1_1_conv2d_0_kernel_size=(3, 3),
                                   module1_1_conv2d_0_stride=(2, 2),
                                   module1_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module1_1_conv2d_0_pad_mode="pad")
        self.module11_1 = Module11(module1_0_conv2d_0_in_channels=128,
                                   module1_0_conv2d_0_out_channels=64,
                                   module1_0_conv2d_0_kernel_size=(1, 1),
                                   module1_0_conv2d_0_stride=(1, 1),
                                   module1_0_conv2d_0_padding=0,
                                   module1_0_conv2d_0_pad_mode="valid",
                                   module1_1_conv2d_0_in_channels=64,
                                   module1_1_conv2d_0_out_channels=64,
                                   module1_1_conv2d_0_kernel_size=(1, 1),
                                   module1_1_conv2d_0_stride=(1, 1),
                                   module1_1_conv2d_0_padding=0,
                                   module1_1_conv2d_0_pad_mode="valid",
                                   module1_2_conv2d_0_in_channels=64,
                                   module1_2_conv2d_0_out_channels=64,
                                   module1_2_conv2d_0_kernel_size=(3, 3),
                                   module1_2_conv2d_0_stride=(1, 1),
                                   module1_2_conv2d_0_padding=(1, 1, 1, 1),
                                   module1_2_conv2d_0_pad_mode="pad")
        self.module12_2 = Module12(module1_0_conv2d_0_in_channels=64,
                                   module1_0_conv2d_0_out_channels=64,
                                   module1_0_conv2d_0_kernel_size=(1, 1),
                                   module1_0_conv2d_0_stride=(1, 1),
                                   module1_0_conv2d_0_padding=0,
                                   module1_0_conv2d_0_pad_mode="valid",
                                   module1_1_conv2d_0_in_channels=64,
                                   module1_1_conv2d_0_out_channels=64,
                                   module1_1_conv2d_0_kernel_size=(3, 3),
                                   module1_1_conv2d_0_stride=(1, 1),
                                   module1_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module1_1_conv2d_0_pad_mode="pad")
        self.module12_3 = Module12(module1_0_conv2d_0_in_channels=64,
                                   module1_0_conv2d_0_out_channels=64,
                                   module1_0_conv2d_0_kernel_size=(1, 1),
                                   module1_0_conv2d_0_stride=(1, 1),
                                   module1_0_conv2d_0_padding=0,
                                   module1_0_conv2d_0_pad_mode="valid",
                                   module1_1_conv2d_0_in_channels=64,
                                   module1_1_conv2d_0_out_channels=64,
                                   module1_1_conv2d_0_kernel_size=(3, 3),
                                   module1_1_conv2d_0_stride=(1, 1),
                                   module1_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module1_1_conv2d_0_pad_mode="pad")
        self.module1_1 = Module1(conv2d_0_in_channels=128,
                                 conv2d_0_out_channels=64,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=0,
                                 conv2d_0_pad_mode="valid")
        self.concat_62 = P.Concat(axis=1)
        self.module1_2 = Module1(conv2d_0_in_channels=128,
                                 conv2d_0_out_channels=128,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=0,
                                 conv2d_0_pad_mode="valid")
        self.module1_3 = Module1(conv2d_0_in_channels=128,
                                 conv2d_0_out_channels=256,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(2, 2),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.module11_2 = Module11(module1_0_conv2d_0_in_channels=256,
                                   module1_0_conv2d_0_out_channels=128,
                                   module1_0_conv2d_0_kernel_size=(1, 1),
                                   module1_0_conv2d_0_stride=(1, 1),
                                   module1_0_conv2d_0_padding=0,
                                   module1_0_conv2d_0_pad_mode="valid",
                                   module1_1_conv2d_0_in_channels=128,
                                   module1_1_conv2d_0_out_channels=128,
                                   module1_1_conv2d_0_kernel_size=(1, 1),
                                   module1_1_conv2d_0_stride=(1, 1),
                                   module1_1_conv2d_0_padding=0,
                                   module1_1_conv2d_0_pad_mode="valid",
                                   module1_2_conv2d_0_in_channels=128,
                                   module1_2_conv2d_0_out_channels=128,
                                   module1_2_conv2d_0_kernel_size=(3, 3),
                                   module1_2_conv2d_0_stride=(1, 1),
                                   module1_2_conv2d_0_padding=(1, 1, 1, 1),
                                   module1_2_conv2d_0_pad_mode="pad")
        self.module12_4 = Module12(module1_0_conv2d_0_in_channels=128,
                                   module1_0_conv2d_0_out_channels=128,
                                   module1_0_conv2d_0_kernel_size=(1, 1),
                                   module1_0_conv2d_0_stride=(1, 1),
                                   module1_0_conv2d_0_padding=0,
                                   module1_0_conv2d_0_pad_mode="valid",
                                   module1_1_conv2d_0_in_channels=128,
                                   module1_1_conv2d_0_out_channels=128,
                                   module1_1_conv2d_0_kernel_size=(3, 3),
                                   module1_1_conv2d_0_stride=(1, 1),
                                   module1_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module1_1_conv2d_0_pad_mode="pad")
        self.module12_5 = Module12(module1_0_conv2d_0_in_channels=128,
                                   module1_0_conv2d_0_out_channels=128,
                                   module1_0_conv2d_0_kernel_size=(1, 1),
                                   module1_0_conv2d_0_stride=(1, 1),
                                   module1_0_conv2d_0_padding=0,
                                   module1_0_conv2d_0_pad_mode="valid",
                                   module1_1_conv2d_0_in_channels=128,
                                   module1_1_conv2d_0_out_channels=128,
                                   module1_1_conv2d_0_kernel_size=(3, 3),
                                   module1_1_conv2d_0_stride=(1, 1),
                                   module1_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module1_1_conv2d_0_pad_mode="pad")
        self.module1_4 = Module1(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=128,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=0,
                                 conv2d_0_pad_mode="valid")
        self.concat_96 = P.Concat(axis=1)
        self.module1_5 = Module1(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=256,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=0,
                                 conv2d_0_pad_mode="valid")
        self.module1_6 = Module1(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=512,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(2, 2),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.module1_7 = Module1(conv2d_0_in_channels=512,
                                 conv2d_0_out_channels=256,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=0,
                                 conv2d_0_pad_mode="valid")
        self.pad_maxpool2d_106 = nn.Pad(paddings=((0, 0), (0, 0), (2, 2), (2, 2)))
        self.maxpool2d_106 = nn.MaxPool2d(kernel_size=(5, 5), stride=(1, 1))
        self.pad_maxpool2d_107 = nn.Pad(paddings=((0, 0), (0, 0), (4, 4), (4, 4)))
        self.maxpool2d_107 = nn.MaxPool2d(kernel_size=(9, 9), stride=(1, 1))
        self.pad_maxpool2d_108 = nn.Pad(paddings=((0, 0), (0, 0), (6, 6), (6, 6)))
        self.maxpool2d_108 = nn.MaxPool2d(kernel_size=(13, 13), stride=(1, 1))
        self.concat_109 = P.Concat(axis=1)
        self.module1_8 = Module1(conv2d_0_in_channels=1024,
                                 conv2d_0_out_channels=512,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=0,
                                 conv2d_0_pad_mode="valid")
        self.module1_9 = Module1(conv2d_0_in_channels=512,
                                 conv2d_0_out_channels=256,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=0,
                                 conv2d_0_pad_mode="valid")
        self.module12_6 = Module12(module1_0_conv2d_0_in_channels=256,
                                   module1_0_conv2d_0_out_channels=256,
                                   module1_0_conv2d_0_kernel_size=(1, 1),
                                   module1_0_conv2d_0_stride=(1, 1),
                                   module1_0_conv2d_0_padding=0,
                                   module1_0_conv2d_0_pad_mode="valid",
                                   module1_1_conv2d_0_in_channels=256,
                                   module1_1_conv2d_0_out_channels=256,
                                   module1_1_conv2d_0_kernel_size=(3, 3),
                                   module1_1_conv2d_0_stride=(1, 1),
                                   module1_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module1_1_conv2d_0_pad_mode="pad")
        self.module1_10 = Module1(conv2d_0_in_channels=512,
                                  conv2d_0_out_channels=256,
                                  conv2d_0_kernel_size=(1, 1),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=0,
                                  conv2d_0_pad_mode="valid")
        self.concat_125 = P.Concat(axis=1)
        self.module12_7 = Module12(module1_0_conv2d_0_in_channels=512,
                                   module1_0_conv2d_0_out_channels=512,
                                   module1_0_conv2d_0_kernel_size=(1, 1),
                                   module1_0_conv2d_0_stride=(1, 1),
                                   module1_0_conv2d_0_padding=0,
                                   module1_0_conv2d_0_pad_mode="valid",
                                   module1_1_conv2d_0_in_channels=512,
                                   module1_1_conv2d_0_out_channels=256,
                                   module1_1_conv2d_0_kernel_size=(1, 1),
                                   module1_1_conv2d_0_stride=(1, 1),
                                   module1_1_conv2d_0_padding=0,
                                   module1_1_conv2d_0_pad_mode="valid")
        self.resizenearestneighbor_132 = P.ResizeNearestNeighbor(size=(img_size//16, img_size//16))
        self.concat_133 = P.Concat(axis=1)
        self.module15_0 = Module15(module1_0_conv2d_0_in_channels=512,
                                   module1_0_conv2d_0_out_channels=128,
                                   module1_0_conv2d_0_kernel_size=(1, 1),
                                   module1_0_conv2d_0_stride=(1, 1),
                                   module1_0_conv2d_0_padding=0,
                                   module1_0_conv2d_0_pad_mode="valid",
                                   module1_1_conv2d_0_in_channels=128,
                                   module1_1_conv2d_0_out_channels=128,
                                   module1_1_conv2d_0_kernel_size=(1, 1),
                                   module1_1_conv2d_0_stride=(1, 1),
                                   module1_1_conv2d_0_padding=0,
                                   module1_1_conv2d_0_pad_mode="valid",
                                   module1_2_conv2d_0_in_channels=128,
                                   module1_2_conv2d_0_out_channels=128,
                                   module1_2_conv2d_0_kernel_size=(3, 3),
                                   module1_2_conv2d_0_stride=(1, 1),
                                   module1_2_conv2d_0_padding=(1, 1, 1, 1),
                                   module1_2_conv2d_0_pad_mode="pad")
        self.module1_11 = Module1(conv2d_0_in_channels=512,
                                  conv2d_0_out_channels=128,
                                  conv2d_0_kernel_size=(1, 1),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=0,
                                  conv2d_0_pad_mode="valid")
        self.concat_146 = P.Concat(axis=1)
        self.module12_8 = Module12(module1_0_conv2d_0_in_channels=256,
                                   module1_0_conv2d_0_out_channels=256,
                                   module1_0_conv2d_0_kernel_size=(1, 1),
                                   module1_0_conv2d_0_stride=(1, 1),
                                   module1_0_conv2d_0_padding=0,
                                   module1_0_conv2d_0_pad_mode="valid",
                                   module1_1_conv2d_0_in_channels=256,
                                   module1_1_conv2d_0_out_channels=128,
                                   module1_1_conv2d_0_kernel_size=(1, 1),
                                   module1_1_conv2d_0_stride=(1, 1),
                                   module1_1_conv2d_0_padding=0,
                                   module1_1_conv2d_0_pad_mode="valid")
        self.resizenearestneighbor_153 = P.ResizeNearestNeighbor(size=(img_size//8, img_size//8))
        self.concat_154 = P.Concat(axis=1)
        self.module15_1 = Module15(module1_0_conv2d_0_in_channels=256,
                                   module1_0_conv2d_0_out_channels=64,
                                   module1_0_conv2d_0_kernel_size=(1, 1),
                                   module1_0_conv2d_0_stride=(1, 1),
                                   module1_0_conv2d_0_padding=0,
                                   module1_0_conv2d_0_pad_mode="valid",
                                   module1_1_conv2d_0_in_channels=64,
                                   module1_1_conv2d_0_out_channels=64,
                                   module1_1_conv2d_0_kernel_size=(1, 1),
                                   module1_1_conv2d_0_stride=(1, 1),
                                   module1_1_conv2d_0_padding=0,
                                   module1_1_conv2d_0_pad_mode="valid",
                                   module1_2_conv2d_0_in_channels=64,
                                   module1_2_conv2d_0_out_channels=64,
                                   module1_2_conv2d_0_kernel_size=(3, 3),
                                   module1_2_conv2d_0_stride=(1, 1),
                                   module1_2_conv2d_0_padding=(1, 1, 1, 1),
                                   module1_2_conv2d_0_pad_mode="pad")
        self.module1_12 = Module1(conv2d_0_in_channels=256,
                                  conv2d_0_out_channels=64,
                                  conv2d_0_kernel_size=(1, 1),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=0,
                                  conv2d_0_pad_mode="valid")
        self.concat_167 = P.Concat(axis=1)
        self.module1_13 = Module1(conv2d_0_in_channels=128,
                                  conv2d_0_out_channels=128,
                                  conv2d_0_kernel_size=(1, 1),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=0,
                                  conv2d_0_pad_mode="valid")
        self.module1_14 = Module1(conv2d_0_in_channels=128,
                                  conv2d_0_out_channels=128,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(2, 2),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.concat_177 = P.Concat(axis=1)
        self.module15_2 = Module15(module1_0_conv2d_0_in_channels=256,
                                   module1_0_conv2d_0_out_channels=128,
                                   module1_0_conv2d_0_kernel_size=(1, 1),
                                   module1_0_conv2d_0_stride=(1, 1),
                                   module1_0_conv2d_0_padding=0,
                                   module1_0_conv2d_0_pad_mode="valid",
                                   module1_1_conv2d_0_in_channels=128,
                                   module1_1_conv2d_0_out_channels=128,
                                   module1_1_conv2d_0_kernel_size=(1, 1),
                                   module1_1_conv2d_0_stride=(1, 1),
                                   module1_1_conv2d_0_padding=0,
                                   module1_1_conv2d_0_pad_mode="valid",
                                   module1_2_conv2d_0_in_channels=128,
                                   module1_2_conv2d_0_out_channels=128,
                                   module1_2_conv2d_0_kernel_size=(3, 3),
                                   module1_2_conv2d_0_stride=(1, 1),
                                   module1_2_conv2d_0_padding=(1, 1, 1, 1),
                                   module1_2_conv2d_0_pad_mode="pad")
        self.module1_15 = Module1(conv2d_0_in_channels=256,
                                  conv2d_0_out_channels=128,
                                  conv2d_0_kernel_size=(1, 1),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=0,
                                  conv2d_0_pad_mode="valid")
        self.concat_203 = P.Concat(axis=1)
        self.module1_16 = Module1(conv2d_0_in_channels=256,
                                  conv2d_0_out_channels=256,
                                  conv2d_0_kernel_size=(1, 1),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=0,
                                  conv2d_0_pad_mode="valid")
        self.module1_17 = Module1(conv2d_0_in_channels=256,
                                  conv2d_0_out_channels=256,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(2, 2),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.concat_213 = P.Concat(axis=1)
        self.module15_3 = Module15(module1_0_conv2d_0_in_channels=512,
                                   module1_0_conv2d_0_out_channels=256,
                                   module1_0_conv2d_0_kernel_size=(1, 1),
                                   module1_0_conv2d_0_stride=(1, 1),
                                   module1_0_conv2d_0_padding=0,
                                   module1_0_conv2d_0_pad_mode="valid",
                                   module1_1_conv2d_0_in_channels=256,
                                   module1_1_conv2d_0_out_channels=256,
                                   module1_1_conv2d_0_kernel_size=(1, 1),
                                   module1_1_conv2d_0_stride=(1, 1),
                                   module1_1_conv2d_0_padding=0,
                                   module1_1_conv2d_0_pad_mode="valid",
                                   module1_2_conv2d_0_in_channels=256,
                                   module1_2_conv2d_0_out_channels=256,
                                   module1_2_conv2d_0_kernel_size=(3, 3),
                                   module1_2_conv2d_0_stride=(1, 1),
                                   module1_2_conv2d_0_padding=(1, 1, 1, 1),
                                   module1_2_conv2d_0_pad_mode="pad")
        self.module1_18 = Module1(conv2d_0_in_channels=512,
                                  conv2d_0_out_channels=256,
                                  conv2d_0_kernel_size=(1, 1),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=0,
                                  conv2d_0_pad_mode="valid")
        self.concat_239 = P.Concat(axis=1)
        self.module1_19 = Module1(conv2d_0_in_channels=512,
                                  conv2d_0_out_channels=512,
                                  conv2d_0_kernel_size=(1, 1),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=0,
                                  conv2d_0_pad_mode="valid")
        self.conv2d_172 = nn.Conv2d(in_channels=128,
                                    out_channels=33,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.reshape_174 = P.Reshape()
        self.reshape_174_shape = tuple([bs, 3, 11, img_size//8, img_size//8])
        self.transpose_176 = P.Transpose()
        self.sigmoid_178 = nn.Sigmoid()
        self.module8_0 = Module8(stridedslice_0_begin=(0, 0, 0, 0, 0), stridedslice_0_end=(bs, 3, img_size//8, img_size//8, 2))
        self.sub_190_bias = 0.5
        self.add_193_bias = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, img_size//8, img_size//8, 2)).astype(np.float32)),
                                      name=None)
        self.mul_196_w = 8.0
        self.module8_1 = Module8(stridedslice_0_begin=(0, 0, 0, 0, 2), stridedslice_0_end=(bs, 3, img_size//8, img_size//8, 4))
        self.pow_191 = P.Pow()
        self.pow_191_input_weight = 2.0
        self.mul_194_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 3, 1, 1, 2)).astype(np.float32)), name=None)
        self.stridedslice_183 = P.StridedSlice()
        self.stridedslice_183_begin = (0, 0, 0, 0, 4)
        self.stridedslice_183_end = (bs, 3, img_size//8, img_size//8, 11)
        self.stridedslice_183_strides = (1, 1, 1, 1, 1)
        self.concat_198 = P.Concat(axis=-1)
        self.reshape_200 = P.Reshape()
        self.reshape_200_shape = tuple([bs, 3*(img_size//8)*(img_size//8), 11])
        self.conv2d_208 = nn.Conv2d(in_channels=256,
                                    out_channels=33,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.reshape_210 = P.Reshape()
        self.reshape_210_shape = tuple([bs, 3, 11, img_size//16, img_size//16])
        self.transpose_212 = P.Transpose()
        self.sigmoid_214 = nn.Sigmoid()
        self.module8_2 = Module8(stridedslice_0_begin=(0, 0, 0, 0, 0), stridedslice_0_end=(bs, 3, img_size//16, img_size//16, 2))
        self.sub_226_bias = 0.5
        self.add_229_bias = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, img_size//16, img_size//16, 2)).astype(np.float32)), name=None)
        self.mul_232_w = 16.0
        self.module8_3 = Module8(stridedslice_0_begin=(0, 0, 0, 0, 2), stridedslice_0_end=(bs, 3, img_size//16, img_size//16, 4))
        self.pow_227 = P.Pow()
        self.pow_227_input_weight = 2.0
        self.mul_230_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 3, 1, 1, 2)).astype(np.float32)), name=None)
        self.stridedslice_219 = P.StridedSlice()
        self.stridedslice_219_begin = (0, 0, 0, 0, 4)
        self.stridedslice_219_end = (bs, 3, img_size//16, img_size//16, 11)
        self.stridedslice_219_strides = (1, 1, 1, 1, 1)
        self.concat_234 = P.Concat(axis=-1)
        self.reshape_236 = P.Reshape()
        self.reshape_236_shape = tuple([bs, 3*(img_size//16)*(img_size//16), 11])
        self.conv2d_243 = nn.Conv2d(in_channels=512,
                                    out_channels=33,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.reshape_244 = P.Reshape()
        self.reshape_244_shape = tuple([bs, 3, 11, img_size//32, img_size//32])
        self.transpose_245 = P.Transpose()
        self.sigmoid_246 = nn.Sigmoid()
        self.module8_4 = Module8(stridedslice_0_begin=(0, 0, 0, 0, 0), stridedslice_0_end=(bs, 3, img_size//32, img_size//32, 2))
        self.sub_252_bias = 0.5
        self.add_254_bias = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, img_size//32, img_size//32, 2)).astype(np.float32)), name=None)
        self.mul_256_w = 32.0
        self.module8_5 = Module8(stridedslice_0_begin=(0, 0, 0, 0, 2), stridedslice_0_end=(bs, 3, img_size//32, img_size//32, 4))
        self.pow_253 = P.Pow()
        self.pow_253_input_weight = 2.0
        self.mul_255_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 3, 1, 1, 2)).astype(np.float32)), name=None)
        self.stridedslice_249 = P.StridedSlice()
        self.stridedslice_249_begin = (0, 0, 0, 0, 4)
        self.stridedslice_249_end = (bs, 3, img_size//32, img_size//32, 11)
        self.stridedslice_249_strides = (1, 1, 1, 1, 1)
        self.concat_257 = P.Concat(axis=-1)
        self.reshape_258 = P.Reshape()
        self.reshape_258_shape = tuple([bs, 3*(img_size//32)*(img_size//32), 11])
        self.concat_259 = P.Concat(axis=1)

    def construct(self, images):
        opt_stridedslice_0 = self.stridedslice_0(images, self.stridedslice_0_begin, self.stridedslice_0_end,
                                                 self.stridedslice_0_strides)
        opt_stridedslice_4 = self.stridedslice_4(opt_stridedslice_0, self.stridedslice_4_begin, self.stridedslice_4_end,
                                                 self.stridedslice_4_strides)
        opt_stridedslice_1 = self.stridedslice_1(images, self.stridedslice_1_begin, self.stridedslice_1_end,
                                                 self.stridedslice_1_strides)
        opt_stridedslice_5 = self.stridedslice_5(opt_stridedslice_1, self.stridedslice_5_begin, self.stridedslice_5_end,
                                                 self.stridedslice_5_strides)
        opt_stridedslice_2 = self.stridedslice_2(images, self.stridedslice_2_begin, self.stridedslice_2_end,
                                                 self.stridedslice_2_strides)
        opt_stridedslice_6 = self.stridedslice_6(opt_stridedslice_2, self.stridedslice_6_begin, self.stridedslice_6_end,
                                                 self.stridedslice_6_strides)
        opt_stridedslice_3 = self.stridedslice_3(images, self.stridedslice_3_begin, self.stridedslice_3_end,
                                                 self.stridedslice_3_strides)
        opt_stridedslice_7 = self.stridedslice_7(opt_stridedslice_3, self.stridedslice_7_begin, self.stridedslice_7_end,
                                                 self.stridedslice_7_strides)
        opt_concat_8 = self.concat_8((opt_stridedslice_4, opt_stridedslice_5, opt_stridedslice_6, opt_stridedslice_7, ))
        module12_0_opt = self.module12_0(opt_concat_8)
        module11_0_opt = self.module11_0(module12_0_opt)
        module1_0_opt = self.module1_0(module12_0_opt)
        opt_concat_28 = self.concat_28((module11_0_opt, module1_0_opt, ))
        module12_1_opt = self.module12_1(opt_concat_28)
        module11_1_opt = self.module11_1(module12_1_opt)
        module12_2_opt = self.module12_2(module11_1_opt)
        opt_add_54 = P.Add()(module11_1_opt, module12_2_opt)
        module12_3_opt = self.module12_3(opt_add_54)
        opt_add_61 = P.Add()(opt_add_54, module12_3_opt)
        module1_1_opt = self.module1_1(module12_1_opt)
        opt_concat_62 = self.concat_62((opt_add_61, module1_1_opt, ))
        module1_2_opt = self.module1_2(opt_concat_62)
        module1_3_opt = self.module1_3(module1_2_opt)
        module11_2_opt = self.module11_2(module1_3_opt)
        module12_4_opt = self.module12_4(module11_2_opt)
        opt_add_88 = P.Add()(module11_2_opt, module12_4_opt)
        module12_5_opt = self.module12_5(opt_add_88)
        opt_add_95 = P.Add()(opt_add_88, module12_5_opt)
        module1_4_opt = self.module1_4(module1_3_opt)
        opt_concat_96 = self.concat_96((opt_add_95, module1_4_opt, ))
        module1_5_opt = self.module1_5(opt_concat_96)
        module1_6_opt = self.module1_6(module1_5_opt)
        module1_7_opt = self.module1_7(module1_6_opt)
        opt_maxpool2d_106 = self.pad_maxpool2d_106(module1_7_opt)
        opt_maxpool2d_106 = self.maxpool2d_106(opt_maxpool2d_106)
        opt_maxpool2d_107 = self.pad_maxpool2d_107(module1_7_opt)
        opt_maxpool2d_107 = self.maxpool2d_107(opt_maxpool2d_107)
        opt_maxpool2d_108 = self.pad_maxpool2d_108(module1_7_opt)
        opt_maxpool2d_108 = self.maxpool2d_108(opt_maxpool2d_108)
        opt_concat_109 = self.concat_109((module1_7_opt, opt_maxpool2d_106, opt_maxpool2d_107, opt_maxpool2d_108, ))
        module1_8_opt = self.module1_8(opt_concat_109)
        module1_9_opt = self.module1_9(module1_8_opt)
        module12_6_opt = self.module12_6(module1_9_opt)
        module1_10_opt = self.module1_10(module1_8_opt)
        opt_concat_125 = self.concat_125((module12_6_opt, module1_10_opt, ))
        module12_7_opt = self.module12_7(opt_concat_125)
        opt_resizenearestneighbor_132 = self.resizenearestneighbor_132(module12_7_opt)
        opt_concat_133 = self.concat_133((opt_resizenearestneighbor_132, module1_5_opt, ))
        module15_0_opt = self.module15_0(opt_concat_133)
        module1_11_opt = self.module1_11(opt_concat_133)
        opt_concat_146 = self.concat_146((module15_0_opt, module1_11_opt, ))
        module12_8_opt = self.module12_8(opt_concat_146)
        opt_resizenearestneighbor_153 = self.resizenearestneighbor_153(module12_8_opt)
        opt_concat_154 = self.concat_154((opt_resizenearestneighbor_153, module1_2_opt, ))
        module15_1_opt = self.module15_1(opt_concat_154)
        module1_12_opt = self.module1_12(opt_concat_154)
        opt_concat_167 = self.concat_167((module15_1_opt, module1_12_opt, ))
        module1_13_opt = self.module1_13(opt_concat_167)
        module1_14_opt = self.module1_14(module1_13_opt)
        opt_concat_177 = self.concat_177((module1_14_opt, module12_8_opt, ))
        module15_2_opt = self.module15_2(opt_concat_177)
        module1_15_opt = self.module1_15(opt_concat_177)
        opt_concat_203 = self.concat_203((module15_2_opt, module1_15_opt, ))
        module1_16_opt = self.module1_16(opt_concat_203)
        module1_17_opt = self.module1_17(module1_16_opt)
        opt_concat_213 = self.concat_213((module1_17_opt, module12_7_opt, ))
        module15_3_opt = self.module15_3(opt_concat_213)
        module1_18_opt = self.module1_18(opt_concat_213)
        opt_concat_239 = self.concat_239((module15_3_opt, module1_18_opt, ))
        module1_19_opt = self.module1_19(opt_concat_239)
        opt_conv2d_172 = self.conv2d_172(module1_13_opt)
        opt_reshape_174 = self.reshape_174(opt_conv2d_172, self.reshape_174_shape)
        opt_transpose_176 = self.transpose_176(opt_reshape_174, (0, 1, 3, 4, 2))
        opt_sigmoid_178 = self.sigmoid_178(opt_transpose_176)
        module8_0_opt = self.module8_0(opt_sigmoid_178)
        opt_sub_190 = module8_0_opt - self.sub_190_bias
        opt_add_193 = opt_sub_190 + self.add_193_bias
        opt_mul_196 = opt_add_193 * self.mul_196_w
        module8_1_opt = self.module8_1(opt_sigmoid_178)
        opt_pow_191 = self.pow_191(module8_1_opt, self.pow_191_input_weight)
        opt_mul_194 = opt_pow_191 * self.mul_194_w
        opt_stridedslice_183 = self.stridedslice_183(opt_sigmoid_178, self.stridedslice_183_begin,
                                                     self.stridedslice_183_end, self.stridedslice_183_strides)
        # small
        opt_concat_198 = self.concat_198((opt_mul_196, opt_mul_194, opt_stridedslice_183, ))
        opt_reshape_200 = self.reshape_200(opt_concat_198, self.reshape_200_shape)

        opt_conv2d_208 = self.conv2d_208(module1_16_opt)
        opt_reshape_210 = self.reshape_210(opt_conv2d_208, self.reshape_210_shape)
        opt_transpose_212 = self.transpose_212(opt_reshape_210, (0, 1, 3, 4, 2))
        opt_sigmoid_214 = self.sigmoid_214(opt_transpose_212)
        module8_2_opt = self.module8_2(opt_sigmoid_214)
        opt_sub_226 = module8_2_opt - self.sub_226_bias
        opt_add_229 = opt_sub_226 + self.add_229_bias
        opt_mul_232 = opt_add_229 * self.mul_232_w
        module8_3_opt = self.module8_3(opt_sigmoid_214)
        opt_pow_227 = self.pow_227(module8_3_opt, self.pow_227_input_weight)
        opt_mul_230 = opt_pow_227 * self.mul_230_w
        opt_stridedslice_219 = self.stridedslice_219(opt_sigmoid_214, self.stridedslice_219_begin,
                                                     self.stridedslice_219_end, self.stridedslice_219_strides)
        # medium
        opt_concat_234 = self.concat_234((opt_mul_232, opt_mul_230, opt_stridedslice_219, ))
        opt_reshape_236 = self.reshape_236(opt_concat_234, self.reshape_236_shape)

        opt_conv2d_243 = self.conv2d_243(module1_19_opt)
        opt_reshape_244 = self.reshape_244(opt_conv2d_243, self.reshape_244_shape)
        opt_transpose_245 = self.transpose_245(opt_reshape_244, (0, 1, 3, 4, 2))
        opt_sigmoid_246 = self.sigmoid_246(opt_transpose_245)
        module8_4_opt = self.module8_4(opt_sigmoid_246)
        opt_sub_252 = module8_4_opt - self.sub_252_bias
        opt_add_254 = opt_sub_252 + self.add_254_bias
        opt_mul_256 = opt_add_254 * self.mul_256_w
        module8_5_opt = self.module8_5(opt_sigmoid_246)
        opt_pow_253 = self.pow_253(module8_5_opt, self.pow_253_input_weight)
        opt_mul_255 = opt_pow_253 * self.mul_255_w
        opt_stridedslice_249 = self.stridedslice_249(opt_sigmoid_246, self.stridedslice_249_begin,
                                                     self.stridedslice_249_end, self.stridedslice_249_strides)
        # large
        opt_concat_257 = self.concat_257((opt_mul_256, opt_mul_255, opt_stridedslice_249, ))
        opt_reshape_258 = self.reshape_258(opt_concat_257, self.reshape_258_shape)
        preds = self.concat_259((opt_reshape_200, opt_reshape_236, opt_reshape_258, ))
        return opt_transpose_176, opt_transpose_212, opt_transpose_245, opt_concat_198[..., :4]/1024., opt_concat_234[..., :4]/1024., opt_concat_257[..., :4]/1024., preds


class YoloWithLossCell(nn.Cell):
    """YOLOV5 loss."""

    def __init__(self, network):
        super(YoloWithLossCell, self).__init__()
        self.yolo_network = network
        self.loss_big = YoloLossBlock('l')
        self.loss_me = YoloLossBlock('m')
        self.loss_small = YoloLossBlock('s')
        self.tenser_to_array = P.TupleToArray()
        self.transpose = P.Transpose()
        self.tp_perm = (0, 2, 3, 1, 4)
    def construct(self, x, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2):
        input_shape = F.shape(x)[2:4]
        input_shape = F.cast(self.tenser_to_array(input_shape) * 2, ms.float32)

        yolo_out = self.yolo_network(x)
        pred_s = self.transpose(yolo_out[0], self.tp_perm)
        pred_m = self.transpose(yolo_out[1], self.tp_perm)
        pred_l = self.transpose(yolo_out[2], self.tp_perm)
        bbox_s = self.transpose(yolo_out[3], self.tp_perm)
        bbox_m = self.transpose(yolo_out[4], self.tp_perm)
        bbox_l = self.transpose(yolo_out[5], self.tp_perm)
        loss_l = self.loss_small(pred_s, bbox_s, y_true_2, gt_2, input_shape)
        loss_m = self.loss_me(pred_m, bbox_m, y_true_1, gt_1, input_shape)
        loss_s = self.loss_big(pred_l, bbox_l, y_true_0, gt_0, input_shape)
        return (loss_l + loss_m + loss_s*0.2)*0.1

class YoloLossBlock(nn.Cell):
    """
    Loss block cell of YOLOV5 network.
    """

    def __init__(self, scale):
        super(YoloLossBlock, self).__init__()
        anchor_scales = np.array([[ 1.25000*8,  1.62500*8],
                                [ 2.00000*8,  3.75000*8],
                                [ 4.12500*8,  2.87500*8],
                                [ 1.87500*16,  3.81250*16],
                                [ 3.87500*16,  2.81250*16],
                                [ 3.68750*16,  7.43750*16],
                                [ 3.62500*32,  2.81250*32],
                                [ 4.87500*32,  6.18750*32],
                                [11.65625*32, 10.18750*32]])
        if scale == 's':
            # anchor mask
            idx = (0, 1, 2)
        elif scale == 'm':
            idx = (3, 4, 5)
        elif scale == 'l':
            idx = (6, 7, 8)
        else:
            raise KeyError("Invalid scale value for DetectionBlock")
        self.anchors = Tensor([anchor_scales[i] for i in idx], ms.float32)
        self.ignore_threshold = Tensor(0.7, ms.float32)
        self.concat = P.Concat(axis=-1)
        self.iou = Iou()
        self.reduce_max = P.ReduceMax(keep_dims=False)
        self.confidence_loss = ConfidenceLoss()
        self.class_loss = ClassLoss()

        self.reduce_sum = P.ReduceSum()
        self.giou = Giou()

    def construct(self, prediction, pred_boxes, y_true, gt_box, input_shape):
        """
        prediction : origin output from yolo
        pred_xy: (sigmoid(xy)+grid)/grid_size
        pred_wh: (exp(wh)*anchors)/input_shape
        y_true : after normalize
        gt_box: [batch, maxboxes, xyhw] after normalize
        """
        object_mask = y_true[:, :, :, :, 4:5]
        class_probs = y_true[:, :, :, :, 5:]
        true_boxes = y_true[:, :, :, :, :4]

        # grid_shape = P.Shape()(prediction)[1:3]
        # grid_shape = P.Cast()(F.tuple_to_array(grid_shape[::-1]), ms.float32)

        true_wh = y_true[:, :, :, :, 2:4]
        true_wh = P.Select()(P.Equal()(true_wh, 0.0),
                             P.Fill()(P.DType()(true_wh),
                                      P.Shape()(true_wh), 1.0),
                             true_wh)
        true_wh = P.Log()(true_wh / self.anchors * input_shape)
        # 2-w*h for large picture, use small scale, since small obj need more precise
        box_loss_scale = 2 - y_true[:, :, :, :, 2:3] * y_true[:, :, :, :, 3:4]

        gt_shape = P.Shape()(gt_box)
        gt_box = P.Reshape()(gt_box, (gt_shape[0], 1, 1, 1, gt_shape[1], gt_shape[2]))

        # add one more dimension for broadcast
        iou = self.iou(P.ExpandDims()(pred_boxes, -2), gt_box)
        # gt_box is x,y,h,w after normalize
        # [batch, grid[0], grid[1], num_anchor, num_gt]
        best_iou = self.reduce_max(iou, -1)
        # [batch, grid[0], grid[1], num_anchor]

        # ignore_mask IOU too small
        ignore_mask = best_iou < self.ignore_threshold
        ignore_mask = P.Cast()(ignore_mask, ms.float32)
        ignore_mask = P.ExpandDims()(ignore_mask, -1)
        # ignore_mask backpro will cause a lot maximunGrad and minimumGrad time consume.
        # so we turn off its gradient
        ignore_mask = F.stop_gradient(ignore_mask)

        confidence_loss = self.confidence_loss(object_mask, prediction[:, :, :, :, 4:5], ignore_mask)
        class_loss = self.class_loss(object_mask, prediction[:, :, :, :, 5:], class_probs)

        object_mask_me = P.Reshape()(object_mask, (-1, 1))  # [8, 72, 72, 3, 1]
        box_loss_scale_me = P.Reshape()(box_loss_scale, (-1, 1))
        pred_boxes_me = xywh2x1y1x2y2(pred_boxes)
        pred_boxes_me = P.Reshape()(pred_boxes_me, (-1, 4))
        true_boxes_me = xywh2x1y1x2y2(true_boxes)
        true_boxes_me = P.Reshape()(true_boxes_me, (-1, 4))
        ciou = self.giou(pred_boxes_me, true_boxes_me)
        ciou_loss = object_mask_me * box_loss_scale_me * (1 - ciou)
        ciou_loss_me = self.reduce_sum(ciou_loss, ())
        loss = ciou_loss_me * 4 + confidence_loss + class_loss
        batch_size = P.Shape()(prediction)[0]
        return loss / batch_size

class TrainingWrapper(nn.Cell):
    """Training wrapper."""

    def __init__(self, network, optimizer, accumulate=1, sens=0.0001):
        super(TrainingWrapper, self).__init__()
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)
        # accumulate grads
        # self.accumulate = accumulate
        # self.iter_cnt = Parameter(Tensor(0, ms.int32), requires_grad=False)
        # self.acc_grads = Parameter(Tensor(0, ms.int32), requires_grad=False)

    def construct(self, imgs, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2):
        # self.iter_cnt += 1
        weights = self.weights
        loss = self.network(imgs, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(imgs, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2, sens)
        # if self.reducer_flag:
        #     grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return loss
# --- IOU Class
class Iou(nn.Cell):
    """Calculate the iou of boxes"""

    def __init__(self):
        super(Iou, self).__init__()
        self.min = P.Minimum()
        self.max = P.Maximum()

    def construct(self, box1, box2):
        """
        box1: pred_box [batch, gx, gy, anchors, 1,      4] ->4: [x_center, y_center, w, h]
        box2: gt_box   [batch, 1,  1,  1,       maxbox, 4]
        convert to topLeft and rightDown
        """
        box1_xy = box1[:, :, :, :, :, :2]
        box1_wh = box1[:, :, :, :, :, 2:4]
        box1_mins = box1_xy - box1_wh / F.scalar_to_array(2.0)  # topLeft
        box1_maxs = box1_xy + box1_wh / F.scalar_to_array(2.0)  # rightDown

        box2_xy = box2[:, :, :, :, :, :2]
        box2_wh = box2[:, :, :, :, :, 2:4]
        box2_mins = box2_xy - box2_wh / F.scalar_to_array(2.0)
        box2_maxs = box2_xy + box2_wh / F.scalar_to_array(2.0)

        intersect_mins = self.max(box1_mins, box2_mins)
        intersect_maxs = self.min(box1_maxs, box2_maxs)
        intersect_wh = self.max(intersect_maxs - intersect_mins, F.scalar_to_array(0.0))
        # P.squeeze: for effiecient slice
        intersect_area = P.Squeeze(-1)(intersect_wh[:, :, :, :, :, 0:1]) * \
                         P.Squeeze(-1)(intersect_wh[:, :, :, :, :, 1:2])
        box1_area = P.Squeeze(-1)(box1_wh[:, :, :, :, :, 0:1]) * P.Squeeze(-1)(box1_wh[:, :, :, :, :, 1:2])
        box2_area = P.Squeeze(-1)(box2_wh[:, :, :, :, :, 0:1]) * P.Squeeze(-1)(box2_wh[:, :, :, :, :, 1:2])
        iou = intersect_area / (box1_area + box2_area - intersect_area)
        # iou : [batch, gx, gy, anchors, maxboxes]
        return iou

class Giou(nn.Cell):

    """Calculating giou"""

    def __init__(self):
        super(Giou, self).__init__()
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.min = P.Minimum()
        self.max = P.Maximum()
        self.concat = P.Concat(axis=1)
        self.mean = P.ReduceMean()
        self.div = P.RealDiv()
        self.eps = 0.000001

    def construct(self, box_p, box_gt):
        """construct method"""
        box_p_area = (box_p[..., 2:3] - box_p[..., 0:1]) * (box_p[..., 3:4] - box_p[..., 1:2])
        box_gt_area = (box_gt[..., 2:3] - box_gt[..., 0:1]) * (box_gt[..., 3:4] - box_gt[..., 1:2])
        x_1 = self.max(box_p[..., 0:1], box_gt[..., 0:1])
        x_2 = self.min(box_p[..., 2:3], box_gt[..., 2:3])
        y_1 = self.max(box_p[..., 1:2], box_gt[..., 1:2])
        y_2 = self.min(box_p[..., 3:4], box_gt[..., 3:4])
        intersection = (y_2 - y_1) * (x_2 - x_1)
        xc_1 = self.min(box_p[..., 0:1], box_gt[..., 0:1])
        xc_2 = self.max(box_p[..., 2:3], box_gt[..., 2:3])
        yc_1 = self.min(box_p[..., 1:2], box_gt[..., 1:2])
        yc_2 = self.max(box_p[..., 3:4], box_gt[..., 3:4])
        c_area = (xc_2 - xc_1) * (yc_2 - yc_1)
        union = box_p_area + box_gt_area - intersection
        union = union + self.eps
        c_area = c_area + self.eps
        iou = self.div(self.cast(intersection, ms.float32), self.cast(union, ms.float32))
        res_mid0 = c_area - union
        res_mid1 = self.div(self.cast(res_mid0, ms.float32), self.cast(c_area, ms.float32))
        giou = iou - res_mid1
        giou = C.clip_by_value(giou, -1.0, 1.0)
        return giou

def xywh2x1y1x2y2(box_xywh):
    boxes_x1 = box_xywh[..., 0:1] - box_xywh[..., 2:3] / 2
    boxes_y1 = box_xywh[..., 1:2] - box_xywh[..., 3:4] / 2
    boxes_x2 = box_xywh[..., 0:1] + box_xywh[..., 2:3] / 2
    boxes_y2 = box_xywh[..., 1:2] + box_xywh[..., 3:4] / 2
    boxes_x1y1x2y2 = P.Concat(-1)((boxes_x1, boxes_y1, boxes_x2, boxes_y2))

    return boxes_x1y1x2y2