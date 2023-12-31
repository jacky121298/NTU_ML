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
            nn.Sequential(
                nn.Conv2d(3, bandwidth[0], 3, 1, 1),
                nn.BatchNorm2d(bandwidth[0]),
                nn.ReLU6(),
                nn.MaxPool2d(2, 2, 0),
            ),
            
            nn.Sequential(
                nn.Conv2d(bandwidth[0], bandwidth[1], 3, 1, 1),
                nn.BatchNorm2d(bandwidth[1]),
                nn.ReLU6(),
                nn.MaxPool2d(2, 2, 0),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[1], bandwidth[2], 3, 1, 1),
                nn.BatchNorm2d(bandwidth[2]),
                nn.ReLU6(),
                nn.MaxPool2d(2, 2, 0),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[2], bandwidth[3], 3, 1, 1),
                nn.BatchNorm2d(bandwidth[3]),
                nn.ReLU6(),
                nn.MaxPool2d(2, 2, 0),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[3], bandwidth[4], 3, 1, 1),
                nn.BatchNorm2d(bandwidth[4]),
                nn.ReLU6(),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[4], bandwidth[5], 3, 1, 1),
                nn.BatchNorm2d(bandwidth[5]),
                nn.ReLU6(),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[5], bandwidth[6], 3, 1, 1),
                nn.BatchNorm2d(bandwidth[6]),
                nn.ReLU6(),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[6], bandwidth[7], 3, 1, 1),
                nn.BatchNorm2d(bandwidth[7]),
                nn.ReLU6(),
            ),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Sequential(
            nn.Linear(bandwidth[7], 11),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)