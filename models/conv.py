import paddle.nn as nn
from lib.modules import SynchronizedBatchNorm2d

def conv_bn(inp, oup, k=3, s=1, BatchNorm2d=SynchronizedBatchNorm2d):
    return nn.Sequential(
        nn.Conv2d(inp, oup, k, s, padding=k//2, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )    

def dep_sep_conv_bn(inp, oup, k=3, s=1, BatchNorm2d=SynchronizedBatchNorm2d):
    return nn.Sequential(
        nn.Conv2d(inp, inp, k, s, padding=k//2, groups=inp, bias=False),
        BatchNorm2d(inp),
        nn.ReLU6(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, padding=0, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

conv = {
    'std_conv': conv_bn,
    'dep_sep_conv': dep_sep_conv_bn
}