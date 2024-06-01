import torch
import torch.nn as nn
class DepthwiseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseConvBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.pointwise_conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out
class ECA(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.gamma = gamma
        self.b = b

    def forward(self, x):
        batch_size, num_channels, H, W = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * (self.gamma * y + self.b)

class Net(nn.Module):
    def __init__(self, n_class,drop_rate):
        super(Net, self).__init__()
        self.conv1 = DepthwiseConvBlock(in_channels=3, out_channels=16, kernel_size=3,stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.eca1 = ECA(16)  # Thêm lớp ECA sau lớp conv1
        self.conv2 = DepthwiseConvBlock(in_channels=16, out_channels=32, kernel_size=3,stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.eca2 = ECA(32)  # Thêm lớp ECA sau lớp conv2
        self.conv3 = DepthwiseConvBlock(in_channels=32, out_channels=64, kernel_size=3,stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.eca3 = ECA(64)  # Thêm lớp ECA sau lớp conv3
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(drop_rate)
        self.fc_input_size = 64 * 15 * 16
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, n_class)

    def forward(self, x):
        skip = x
        x = self.pool1(torch.relu(self.bn1(self.eca1(self.conv1(x)))))
        x = self.pool2(torch.relu(self.bn2(self.eca2(self.conv2(x)))))
        x = self.pool3(torch.relu(self.bn3(self.eca3(self.conv3(x)))))
        x = x.view(-1, self.fc_input_size)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# __main__ required for testing with input 124*360
if __name__ == "__main__":
  # Initialize model
  model = Net(n_class=50,drop_rate=0.2)
  input_data = torch.randn(10, 3, 124, 130)
  output = model(input_data)