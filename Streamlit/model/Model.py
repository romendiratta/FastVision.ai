import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(p=0.2)

        self.flatten = nn.Flatten()

        self.fc = nn.Linear(256 * 16 * 16 * 16, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = self.relu(self.conv3(x))
        x = self.pool(x)

        x = self.relu(self.conv4(x))
        x = self.pool(x)

        x = self.flatten(x)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.sigmoid(x)

        return x