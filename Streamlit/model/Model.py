import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool3d(kernel_size=8, stride=8)

        self.dropout = nn.Dropout(p=0.2)

        self.fc = nn.Linear(524288, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)

#         x = self.relu(self.conv2(x))
#         x = self.pool(x)

#         x = self.relu(self.conv3(x))
#         x = self.pool(x)

        x = self.flatten(x)

        x = self.fc(x)
        x = self.dropout(x)
        x = self.sigmoid(x)

        return x