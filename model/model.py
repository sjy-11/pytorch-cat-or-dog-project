import torch.nn as nn
import torch.nn.init as init 

#basic CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.elu1 = nn.ELU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.25)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.elu2 = nn.ELU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=0.25)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.elu3 = nn.ELU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(p=0.25)

        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(128*16*16, 2)

        init.kaiming_uniform_(self.conv1.weight, nonlinearity='leaky_relu')
        init.kaiming_uniform_(self.conv2.weight, nonlinearity='leaky_relu')
        init.kaiming_uniform_(self.conv3.weight, nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.pool1(self.elu1((self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool2(self.elu2(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool3(self.elu3(self.conv3(x)))
        x = self.dropout3(x)
        x = self.fc1(self.flatten1(x))

        return x