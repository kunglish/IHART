import torch
from torch import nn


class LeNet1(nn.Module):
    def __init__(self):
        super(LeNet1, self).__init__()

        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.sigmoid1 = nn.Sigmoid()

        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.sigmoid2 = nn.Sigmoid()

        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.f5 = nn.Linear(16 * 5 * 5, 120)
        self.output = nn.Linear(120, 10)

    def forward(self, x):
        x = self.sigmoid1(self.c1(x))
        x = self.s2(x)
        x = self.sigmoid2(self.c3(x))
        x = self.s4(x)
        x = self.flatten(x)
        x = self.f5(x)
        x = self.output(x)
        return x


if __name__ == "__main__":
    x = torch.rand([1, 1, 28, 28])
    model = LeNet1()
    y = model(x)
    print(y)