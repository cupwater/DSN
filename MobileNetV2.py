import torch
import torch.nn as nn
import math
from collections import OrderedDict
from torchE.nn import SyncBatchNorm2d


BN = None

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        BN(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        BN(oup),
        nn.ReLU(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        # added by gaowei
        self.blockname = None

        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            BN(inp * expand_ratio),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            BN(inp * expand_ratio),
            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            BN(oup),
        )
        self.names = ['0', '1', '2', '3', '4', '5', '6', '7']


    def forward(self, x):
        #from .mobilenet import activations
        #from .mobilenet import train_state

        t = x
        #for layer, (name, module) in enumerate(self.conv._modules.items()):
        for name in self.names:
            module = self.conv._modules[name]
            x = module(x)
            #if isinstance(module, nn.Conv2d) and name in ['3'] and not train_state:
            #    x.retain_grad()
            #    if self.blockname is not None:
            #        activations[self.blockname + '.conv.' + name] = x
            #    else:
            #        activations[name] = x

        if self.use_res_connect:
            return t + x
        else:
            return x


class MobileNetV2(nn.Module):
    #def __init__(self, T, feature_dim, input_size=224, width_mult=1.):
    def __init__(self, T, feature_dim, group_size, group, sync_stats, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        #setting of sync BN
        global BN

        def BNFunc(*args, **kwargs):
            return nn.BatchNorm2d(*args, **kwargs, eps=1e-5, momentum=0.05)
            #return SyncBatchNorm2d(*args, **kwargs, group_size=group_size, group=group, sync_stats=sync_stats, eps=1e-10, momentum=0.05)
        BN = BNFunc
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [T, 24, 2, 2],
            [T, 32, 3, 2],
            [T, 64, 4, 2],
            [T, 96, 3, 1],
            [T, 160, 3, 2],
            [T, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features.append(nn.AvgPool2d(input_size//32, ceil_mode=True))  # such that easily converted to caffemodel
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(self.last_channel, feature_dim),
        )

        self._initialize_weights()

        #for mod in self.modules():
        #    if isinstance(mod, nn.BatchNorm2d):
        #        self._freeze_module(mod)

    def _freeze_module(self, module):
        for p in module.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, SyncBatchNorm2d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()

def mobilenet_v2(load_pretrain=True, T=6, feature_dim=128, group_size=1, group=None, sync_stats=False,input_size=224, width_mult=1.):
    #model = MobileNetV2(T, feature_dim, input_size, width_mult)
    model = MobileNetV2(T, feature_dim, group_size, group, sync_stats)
    if load_pretrain:
        if T == 6:  #load imagenet-pretrained
            state_dict = torch.load('/mnt/lustre/jinxiao/expr/e-resnet/multitask/experiment/imagenet_pretrain/mobilenetv2_t6/mobilenetv2_718.pth.tar')  
            state_dict_copy = OrderedDict([('.'.join(k.split('.')[1:]) ,v) for k, v in state_dict.items() if 'classifier' not in k ])
            model.load_state_dict(state_dict_copy, strict = False)
        else: print('no imagenet-pretrained model found, will train from scatch or load common-pretrained model later')

    return model

