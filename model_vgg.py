import torch.utils.model_zoo as model_zoo
from torch import nn
import torch as torch
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

            nn.Conv2d(512, 384, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 192, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, self.hyperInChn, 1, padding=(0, 0)),
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
        self.model = vgg_backbone(pretrained=pretrained)
        self.lda1_pool = nn.Sequential(
            nn.Conv2d(128, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda1_fc = nn.Linear(16 * 64*4, lda_out_channels)

        self.lda2_pool = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda2_fc = nn.Linear(32 * 16*4, lda_out_channels)

        self.lda3_pool = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda3_fc = nn.Linear(64 * 4*4, lda_out_channels)

        self.lda4_pool = nn.AvgPool2d(7, stride=7)
        self.lda4_fc = nn.Linear(512*4, in_chn - lda_out_channels * 3)

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


class Vgg16Backbone(nn.Module):
    def __init__(self):
        super(Vgg16Backbone, self).__init__()
        self.conv0 = vgg([(2, 64)], 3)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = vgg([(2, 128)], 64)
        self.conv2 = vgg([(3, 256)], 128)
        self.conv3 = vgg([(3, 512)], 256)
        self.conv4 = vgg([(3, 512)], 512)

    def forward(self, img):
        x0 = self.conv0(img)
        x1 = self.conv1(self.maxpool(x0))
        x2 = self.conv2(self.maxpool(x1))
        x3 = self.conv3(self.maxpool(x2))
        x4 = self.conv4(self.maxpool(x3))

        return x1, x2, x3, x4


def vgg_backbone(pretrained=False):
    model = Vgg16Backbone()

    if pretrained:
        save_model = model_zoo.load_url(model_urls['vgg16'])
        save_model = {k: v for k, v in save_model.items()}
        model_dict = model.state_dict()
        keys_save_model = list(save_model.keys())
        keys_model_dict = list(model_dict.keys())

        for i in range(len(keys_model_dict)):
            model_dict[keys_model_dict[i]] = save_model[keys_save_model[i]]

        model.load_state_dict(model_dict)
    else:
        model.apply(weights_init_xavier)
    return model


def vgg(conv_arch, in_channels):
    conv_blks = []
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(*conv_blks)
    # , nn.Flatten(),
    # # 全连接层部分
    # nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
    # nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
    # nn.Linear(4096, 1000)


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for i in range(num_convs):
        if i == 3:
            layers.append(nn.Conv2d(in_channels, out_channels,
                                    kernel_size=1, padding=1))
        else:
            layers.append(nn.Conv2d(in_channels, out_channels,
                                    kernel_size=3, padding=1))
        layers.append(nn.ReLU())

        in_channels = out_channels
    return nn.Sequential(*layers)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    # if isinstance(m, nn.Conv2d):
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
