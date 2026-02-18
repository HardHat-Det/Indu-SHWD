import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class PCRC(nn.Module):
    def __init__(self):
        super().__init__()
        self.C1 = nn.Conv2d(256, 896, kernel_size=1)  # 128+256+512
        # self.R1 = nn.Upsample(None, 2, 'nearest')  # 上采样扩充2倍采用邻近扩充
        self.mcrc = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(896, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 896, kernel_size=1),
        )
        self.acrc = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(896, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 896, kernel_size=1),
        )

    def forward(self, x):
        x1 = self.C1(x)
        x2 = self.mcrc(x1)

        x3 = self.acrc(x1)
        return x2 + x3


# FCN模块实现输入x为tensor列表形式
class FCN(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        super().__init__()
        self.R1 = nn.Upsample(None, 2, 'nearest')  # 上采样扩充2倍采用邻近扩充
        self.R3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 下采样使用最大池化
        self.adjust_0 = nn.Conv2d(512, 256, kernel_size=1)
        self.adjust_2 = nn.Conv2d(128, 256, kernel_size=1)
        self.C1 = nn.Conv2d(256, 3, kernel_size=1)

        # self.C2 = nn.Conv2d(256, 512, kernel_size=1, stride=1)
        # self.C3 = nn.Conv2d(256, 128, kernel_size=1, stride=1)
        self.C4 = nn.Conv2d(1, 128, kernel_size=1, stride=1)
        self.C5 = nn.Conv2d(1, 256, kernel_size=1, stride=1)
        self.C6 = nn.Conv2d(1, 512, kernel_size=1, stride=1)
        self.sig = nn.Sigmoid()
        self.pcrc = PCRC()

    def forward(self, x):
        x0_tmp = x[0]
        x0 = self.R1(x[0])
        # print(x0_tmp.size())
        x0 = self.adjust_0(x0)
        x2_tmp = x[2]
        x2 = self.R3(x[2])
        # print(x2_tmp.size())
        x2 = self.adjust_2(x2)
        x_tmp = torch.add(x0, x[1])
        x1 = torch.add(x_tmp, x2)
        x1_spatial = self.C1(x1)

        Conv_1_1 = torch.split(x1_spatial, 1, dim=1)  # 第一维度1为步长进行分割,spatial attention
        split_list = [512, 256, 128]
        Conv_1_2 = torch.split(self.pcrc(x1), split_list, 1)  # 第一维度256为步长进行分割
        # print(Conv_1_2[1].size())
        # print(self.C6(Conv_1_1[0]).size())
        # print(self.C2(Conv_1_2[0]).size())

        y0 = (x0_tmp * self.sig(Conv_1_2[0]) * torch.sigmoid(self.R3(Conv_1_1[0])))  # y0
        # print(y0.size())
        # y0 = (x0_tmp * self.C2(Conv_1_2[0]) + x0_tmp * self.C6(Conv_1_1[0]))
        # print(self.C5(Conv_1_1[1].size()))
        # print(x[1].size())
        # print(Conv_1_2[1].size)
        y1 = (x[1] * self.sig(Conv_1_2[1])) * self.C5(Conv_1_1[1])  # y1
        # print(y1.size())
        # y1 = (x[1] * Conv_1_2[1]) + (x[1] * self.C5(Conv_1_1[1]))
        # print(self.C4(Conv_1_1[2]).size())
        # print(self.C3(Conv_1_2[2]).size())
        y2 = (x2_tmp * self.sig(Conv_1_2[2])) * (torch.sigmoid(self.R1(Conv_1_1[2])))  # y2
        # print(y2.size())
        # y2 = (self.C3(Conv_1_2[2]) * x2_tmp) + (x2_tmp * self.C4(Conv_1_1[2]))
        # y0 = self.R3(y0)

        # y2 = self.R1(y2)

        return [y0, y1, y2]


if __name__ == '__main__':
    model = FCN()
    model.cuda()

    img1 = torch.rand(1, 512, 20, 20)  # 假设输入1张1024*20*20的特征图
    img2 = torch.rand(1, 256, 40, 40)  # 假设输入1张512*40*40的特征图
    img3 = torch.rand(1, 128, 80, 80)  # 假设输入1张256*80*80的特征图
    img1 = img1.cuda()
    img2 = img2.cuda()
    img3 = img3.cuda()

    with SummaryWriter(comment='FRM') as w:
        w.add_graph(model, ([img1, img2, img3],))
