import torch.nn as nn
import torch.nn.functional as F
import torch

class StudentNet(nn.Module):
    def __init__(self, base = 16, width_mult = 1):
        super(StudentNet, self).__init__()
        multiplier = [1, 2, 4, 8, 16, 16, 16, 16]
        bandwidth = [base * m for m in multiplier]

        # 我們只 Pruning 第三層以後的 Layer
        for i in range(3, 7):
            bandwidth[i] = int(bandwidth[i] * width_mult)

        self.cnn = nn.Sequential(
            # 第一層我們通常不會拆解Convolution Layer。
            nn.Sequential(
                nn.Conv2d(3, bandwidth[0], 3, 1, 1),
                nn.BatchNorm2d(bandwidth[0]),
                nn.ReLU6(),
                nn.MaxPool2d(2, 2, 0),
            ),
            # 接下來每一個Sequential Block都一樣，所以我們只講一個Block
            nn.Sequential(
                # Depthwise Convolution
                nn.Conv2d(bandwidth[0], bandwidth[0], 3, 1, 1, groups = bandwidth[0]),
                # Batch Normalization
                nn.BatchNorm2d(bandwidth[0]),
                # ReLU6 是限制 Neuron 最小只會到 0，最大只會到 6。MobileNet 系列都是使用 ReLU6。
                # 使用 ReLU6 的原因是因為如果數字太大，會不好壓到 float16 or further qunatization，因此才給個限制。
                nn.ReLU6(),
                # Pointwise Convolution
                nn.Conv2d(bandwidth[0], bandwidth[1], 1),
                # 過完 Pointwise Convolution 不需要再做 ReLU，經驗上 Pointwise + ReLU 效果都會變差。
                nn.MaxPool2d(2, 2, 0),
                # 每過完一個 Block 就 Down Sampling
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[1], bandwidth[1], 3, 1, 1, groups = bandwidth[1]),
                nn.BatchNorm2d(bandwidth[1]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[1], bandwidth[2], 1),
                nn.MaxPool2d(2, 2, 0),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[2], bandwidth[2], 3, 1, 1, groups = bandwidth[2]),
                nn.BatchNorm2d(bandwidth[2]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[2], bandwidth[3], 1),
                nn.MaxPool2d(2, 2, 0),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[3], bandwidth[3], 3, 1, 1, groups = bandwidth[3]),
                nn.BatchNorm2d(bandwidth[3]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[3], bandwidth[4], 1),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[4], bandwidth[4], 3, 1, 1, groups = bandwidth[4]),
                nn.BatchNorm2d(bandwidth[4]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[4], bandwidth[5], 1),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[5], bandwidth[5], 3, 1, 1, groups = bandwidth[5]),
                nn.BatchNorm2d(bandwidth[5]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[5], bandwidth[6], 1),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[6], bandwidth[6], 3, 1, 1, groups = bandwidth[6]),
                nn.BatchNorm2d(bandwidth[6]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[6], bandwidth[7], 1),
            ),

            # 這邊我們採用Global Average Pooling。
            # 如果輸入圖片大小不一樣的話，就會因為Global Average Pooling壓成一樣的形狀，這樣子接下來做FC就不會對不起來。
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(bandwidth[7], 11),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)