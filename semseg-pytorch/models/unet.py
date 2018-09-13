import torch.nn as nn

from models.utils import *

class unet(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=1, is_batchnorm=True):
        super(unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        # self.modepool1 = modePool2()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        # self.modepool2 = modePool2()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        # self.modepool3 = modePool2()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        # self.modepool4 = modePool2()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)


    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        # modepool1 = self.modepool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        # modepool2 = self.modepool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        # modepool3 = self.modepool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        # modepool4 = self.modepool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final


def debugUnet():
    model = unet()
    print(model)

    # for param in model.parameters():
    #     print(type(param.data), param.size())
        # print(list(param.data))

    # print(model.state_dict().keys())

    # for key in model.state_dict():
    #     print(key, 'corresponds to', list(model.state_dict()[key]))

    # save model
    # - 保存整个神经网络的结构信息和模型参数信息，save的对象是网络net
    # - 保存神经网络的训练模型参数，save的对象是net.state_dict()

    # torch.save(model, '../debug/net.pkl')
    # torch.save(model.state_dict(), '../debug/net_params.pkl')

    # load model
    # - 对应第一种完整网络结构信息，重载的时候通过torch.load('.pth')直接初始化心得神经网络对象即可
    # - 对应第二种只保存模型参数信息，需要首先导入对应的网络，通过net.load_state_dict(torch.load('.pth'))完成模型参数的重载。
    # net = torch.load('../debug/net.pkl')

debugUnet()
