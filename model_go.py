import torch.nn as nn
import torch
import torch.nn.functional as F

import torch.utils.model_zoo as model_zoo
from torch import nn
import torch as torch
from torch.nn import functional as F
from torch.nn import init
import math

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'google': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'
}


class HyperNet(nn.Module):
    def __init__(self, lda_out_channels, hyper_in_channels, target_in_size, target_fc1_size, target_fc2_size, target_fc3_size, target_fc4_size, feature_size, pretrained=False):
        super(HyperNet, self).__init__()

        self.hyperInChn = hyper_in_channels
        self.target_in_size = target_in_size
        self.f1 = target_fc1_size
        self.f2 = target_fc2_size
        self.f3 = target_fc3_size
        self.f4 = target_fc4_size
        self.feature_size = feature_size

        self.res = LDA(lda_out_channels, target_in_size, pretrained)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv1 = nn.Sequential(

            nn.Conv2d(1024, 512, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.hyperInChn, 1, padding=(0, 0)),
            nn.ReLU(inplace=True)
        )

        self.fc1w_conv = nn.Conv2d(self.hyperInChn, int(self.target_in_size * self.f1 / self.feature_size ** 2), 3,
                                   padding=(1, 1))
        self.fc1b_fc = nn.Linear(self.hyperInChn, self.f1)

        self.fc2w_conv = nn.Conv2d(self.hyperInChn, int(
            self.f1 * self.f2 / self.feature_size ** 2), 3, padding=(1, 1))
        self.fc2b_fc = nn.Linear(self.hyperInChn, self.f2)

        self.fc3w_conv = nn.Conv2d(self.hyperInChn, int(
            self.f2 * self.f3 / self.feature_size ** 2), 3, padding=(1, 1))
        self.fc3b_fc = nn.Linear(self.hyperInChn, self.f3)

        self.fc4w_conv = nn.Conv2d(self.hyperInChn, int(
            self.f3 * self.f4 / self.feature_size ** 2), 3, padding=(1, 1))
        self.fc4b_fc = nn.Linear(self.hyperInChn, self.f4)

        self.fc5w_fc = nn.Linear(self.hyperInChn, self.f4)
        self.fc5b_fc = nn.Linear(self.hyperInChn, 1)

        # initialize
        for i, m_name in enumerate(self._modules):
            if i > 2:
                nn.init.kaiming_normal_(self._modules[m_name].weight.data)

    def forward(self, img):
        feature_size = self.feature_size

        res_out = self.res(img)

        # input vector for target net
        target_in_vec = res_out['target_in_vec'].view(
            -1, self.target_in_size, 1, 1)

        # input features for hyper net
        hyper_in_feat = self.conv1(
            res_out['hyper_in_feat']).view(-1, self.hyperInChn, feature_size, feature_size)

        # generating target net weights & biases
        target_fc1w = self.fc1w_conv(
            hyper_in_feat).view(-1, self.f1, self.target_in_size, 1, 1)
        target_fc1b = self.fc1b_fc(
            self.pool(hyper_in_feat).squeeze()).view(-1, self.f1)

        target_fc2w = self.fc2w_conv(
            hyper_in_feat).view(-1, self.f2, self.f1, 1, 1)
        target_fc2b = self.fc2b_fc(
            self.pool(hyper_in_feat).squeeze()).view(-1, self.f2)

        target_fc3w = self.fc3w_conv(
            hyper_in_feat).view(-1, self.f3, self.f2, 1, 1)
        target_fc3b = self.fc3b_fc(
            self.pool(hyper_in_feat).squeeze()).view(-1, self.f3)

        target_fc4w = self.fc4w_conv(
            hyper_in_feat).view(-1, self.f4, self.f3, 1, 1)
        target_fc4b = self.fc4b_fc(
            self.pool(hyper_in_feat).squeeze()).view(-1, self.f4)

        target_fc5w = self.fc5w_fc(
            self.pool(hyper_in_feat).squeeze()).view(-1, 1, self.f4, 1, 1)
        target_fc5b = self.fc5b_fc(
            self.pool(hyper_in_feat).squeeze()).view(-1, 1)

        out = {}
        out['target_in_vec'] = target_in_vec
        out['target_fc1w'] = target_fc1w
        out['target_fc1b'] = target_fc1b
        out['target_fc2w'] = target_fc2w
        out['target_fc2b'] = target_fc2b
        out['target_fc3w'] = target_fc3w
        out['target_fc3b'] = target_fc3b
        out['target_fc4w'] = target_fc4w
        out['target_fc4b'] = target_fc4b
        out['target_fc5w'] = target_fc5w
        out['target_fc5b'] = target_fc5b

        return out


class LDA(nn.Module):
    def __init__(self, lda_out_channels, in_chn, pretrained=False):
        super(LDA, self).__init__()
        self.model = google_backbone(pretrained=pretrained)
        self.lda1_pool = nn.Sequential(
            nn.Conv2d(192, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda1_fc = nn.Linear(16 * 64, lda_out_channels)

        self.lda2_pool = nn.Sequential(
            nn.Conv2d(480, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda2_fc = nn.Linear(32 * 16, lda_out_channels)

        self.lda3_pool = nn.Sequential(
            nn.Conv2d(832, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda3_fc = nn.Linear(64 * 4, lda_out_channels)

        self.lda4_pool = nn.AvgPool2d(7, stride=7)
        self.lda4_fc = nn.Linear(1024, in_chn - lda_out_channels * 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # initialize
        nn.init.kaiming_normal_(self.lda1_pool._modules['0'].weight.data)
        nn.init.kaiming_normal_(self.lda2_pool._modules['0'].weight.data)
        nn.init.kaiming_normal_(self.lda3_pool._modules['0'].weight.data)
        nn.init.kaiming_normal_(self.lda1_fc.weight.data)
        nn.init.kaiming_normal_(self.lda2_fc.weight.data)
        nn.init.kaiming_normal_(self.lda3_fc.weight.data)
        nn.init.kaiming_normal_(self.lda4_fc.weight.data)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, img):
        data = self.model(img)
        x = data[0]
        lda_1 = self.lda1_fc(self.lda1_pool(x).view(x.size(0), -1))
        x = data[1]
        lda_2 = self.lda2_fc(self.lda2_pool(x).view(x.size(0), -1))
        x = data[2]
        lda_3 = self.lda3_fc(self.lda3_pool(x).view(x.size(0), -1))
        x = data[3]
        lda_4 = self.lda4_fc(self.lda4_pool(x).view(x.size(0), -1))

        vec = torch.cat((lda_1, lda_2, lda_3, lda_4), 1)

        out = {}
        out['hyper_in_feat'] = data[3]
        out['target_in_vec'] = vec

        return out


def google_backbone(pretrained=False):
    model = GoogLeNet()

    if pretrained:
        save_model = model_zoo.load_url(model_urls['google'])
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in save_model.items()
                      if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

    return model






class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(3, 64, kernel_size=7,
                                 stride=2, padding=3)  # BasicConv2d类
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(
            192, 64, 96, 128, 16, 32, 32)           # Inception类
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)


        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = nn.Dropout(0.2)
        # self.fc = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()

    # 正向传播
    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)    # N x 64 x 112 x 112
        x = self.maxpool1(x)  # N x 64 x 56 x 56

        x = self.conv2(x)    # N x 64 x 56 x 56
        x1 = self.conv3(x)    # N x 192 x 56 x 56
        x = self.maxpool2(x1)  # N x 192 x 28 x 28

        x = self.inception3a(x)  # N x 256 x 28 x 28
        x2 = self.inception3b(x)  # N x 480 x 28 x 28
        x = self.maxpool3(x2)     # N x 480 x 14 x 14

        x = self.inception4a(x)  # N x 512 x 14 x 14
        # if self.training and self.aux_logits:  # eval model不执行该部分
        #     aux1 = self.aux1(x)
        x = self.inception4b(x)  # N x 512 x 14 x 14
        x = self.inception4c(x)  # N x 512 x 14 x 14
        x = self.inception4d(x)  # N x 528 x 14 x 14
        # if self.training and self.aux_logits:  # eval model不执行该部分
        #     aux2 = self.aux2(x)
        x3 = self.inception4e(x)  # N x 832 x 14 x 14
        x = self.maxpool4(x3)     # N x 832 x 7 x 7

        x = self.inception5a(x)  # N x 832 x 7 x 7
        x4 = self.inception5b(x)  # N x 1024 x 7 x 7

        # x = self.avgpool(x)      # N x 1024 x 1 x 1
        # x = torch.flatten(x, 1)  # N x 1024
        # x = self.dropout(x)
        # x = self.fc(x)           # N x 1000 (num_classes)

        # if self.training and self.aux_logits:  # eval model不执行该部分
        #     return x, aux2, aux1
        return x1, x2, x3, x4

    # 初始化权重
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# 类Inception，有四个分支


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3,
                        padding=1)   # 保证输出大小等于输入大小
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=3,
                        padding=1)   # 保证输出大小等于输入大小
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1,ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        # 四个分支连接起来
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

# 辅助分类器：类InceptionAux，包括avepool+conv+fc1+fc2



# 类BasicConv2d，包括conv+relu


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x