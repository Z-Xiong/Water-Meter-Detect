import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class YOLOModel(BaseModel):
    def __init__(self, ):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 5, padding=2, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.max_pool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(64, 192, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(192)
        self.max_pool2 = nn.MaxPool2d(2, stride=2)

        self.conv3 = nn.Conv2d(192, 128, 1, padding=0)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, 1, padding=0)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.max_pool3 = nn.MaxPool2d(2, stride=2)

        self.conv7 = nn.Conv2d(512, 256, 1, padding=0)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512, 256, 1, padding=0)
        self.bn9 = nn.BatchNorm2d(256)
        self.conv10 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn10 = nn.BatchNorm2d(512)
        self.conv11 = nn.Conv2d(512, 256, 1, padding=0)
        self.bn11 = nn.BatchNorm2d(256)
        self.conv12 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(512, 256, 1, padding=0)
        self.bn13 = nn.BatchNorm2d(256)
        self.conv14 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn14 = nn.BatchNorm2d(512)
        self.conv15 = nn.Conv2d(512, 512, 1, padding=0)
        self.bn15 = nn.BatchNorm2d(512)
        self.conv16 = nn.Conv2d(512, 1024, 3, padding=1)
        self.bn16 = nn.BatchNorm2d(1024)
        self.max_pool4 = nn.MaxPool2d(2, stride=2)

        self.conv17 = nn.Conv2d(1024, 512, 1, padding=0)
        self.bn17 = nn.BatchNorm2d(512)
        self.conv18 = nn.Conv2d(512, 1024, 3, padding=1)
        self.bn18 = nn.BatchNorm2d(1024)
        self.conv19 = nn.Conv2d(1024, 512, 1, padding=0)
        self.bn19 = nn.BatchNorm2d(512)
        self.conv20 = nn.Conv2d(512, 1024, 3, padding=1)
        self.bn20 = nn.BatchNorm2d(1024)
        self.conv21 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn21 = nn.BatchNorm2d(1024)
        self.conv22 = nn.Conv2d(1024, 1024, 3, padding=1, stride=2)
        self.bn22 = nn.BatchNorm2d(1024)

        self.conv23 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn23 = nn.BatchNorm2d(1024)
        self.conv24 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn24 = nn.BatchNorm2d(1024)

        self.local = nn.Conv2d(1024, 256, 3, padding=1)

        self.fc1 = nn.Linear(12544, 4096)
        self.fc2 = nn.Linear(4096, 1470)

        self.leakyReLU = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leakyReLU(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leakyReLU(x)
        x = self.max_pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leakyReLU(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leakyReLU(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.leakyReLU(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.leakyReLU(x)
        x = self.max_pool3(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.leakyReLU(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.leakyReLU(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = self.leakyReLU(x)
        x = self.conv10(x)
        x = self.bn10(x)
        x = self.leakyReLU(x)
        x = self.conv11(x)
        x = self.bn11(x)
        x = self.leakyReLU(x)
        x = self.conv12(x)
        x = self.bn12(x)
        x = self.leakyReLU(x)
        x = self.conv13(x)
        x = self.bn13(x)
        x = self.leakyReLU(x)
        x = self.conv14(x)
        x = self.bn14(x)
        x = self.leakyReLU(x)
        x = self.conv15(x)
        x = self.bn15(x)
        x = self.leakyReLU(x)
        x = self.conv16(x)
        x = self.bn16(x)
        x = self.leakyReLU(x)
        x = self.max_pool4(x)

        x = self.conv17(x)
        x = self.bn17(x)
        x = self.leakyReLU(x)
        x = self.conv18(x)
        x = self.bn18(x)
        x = self.leakyReLU(x)
        x = self.conv19(x)
        x = self.bn19(x)
        x = self.leakyReLU(x)
        x = self.conv20(x)
        x = self.bn20(x)
        x = self.leakyReLU(x)
        x = self.conv21(x)
        x = self.bn21(x)
        x = self.leakyReLU(x)
        x = self.conv22(x)
        x = self.bn22(x)
        x = self.leakyReLU(x)

        x = self.conv23(x)
        x = self.bn23(x)
        x = self.leakyReLU(x)
        x = self.conv24(x)
        x = self.bn24(x)
        x = self.leakyReLU(x)

        x = self.local(x)
        x = self.leakyReLU(x)

        x = x.view(x.shape[0], -1)  #把向量铺平为全连接层做准备
        x = self.fc1(x)  # 这里的输入为（N，7*7*256）
        x = self.dropout(x)
        x = self.fc2(x)  # [N, 1470]

        x = x.view(x.shape[0], 7, 7, 30)

        return x


def test():
    import torch
    model = YOLOModel()
    img = torch.rand(2,3,448,448)
    output = model(img)
    print(output.size())


if __name__ == '__main__':
    test()