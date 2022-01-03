import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_shape, outputs):
        """
        :param input_shape: (tuple, [3,]) frames x width x height.
        :param outputs: (int) the capacity of the action space.
        """
        super(DQN, self).__init__()
        assert len(input_shape) == 3

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        frame, width, height = input_shape

        # ------------------------------------------------------------------- #
        # Original settings from Pytorch official tutorial
        # self.conv1 = nn.Conv2d(frame, 16, kernel_size=5, stride=2)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        # self.bn3 = nn.BatchNorm2d(32)

        # convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(width)))
        # convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(height)))
        # linear_input_size = convw * convh * 32

        # -------------------------------------------------------------------- #
        # Settings from Gsurma's implementation
        self.conv1 = nn.Conv2d(frame, 32, kernel_size=8, stride=4)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(64)

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(width, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(height, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        self.head_inter = nn.Linear(linear_input_size, 512)
        self.head_output = nn.Linear(512, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # ----------------------------------- #
        # Settings from Pytorch tutorial
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        # return self.head(x.view(x.size(0), -1))

        # ----------------------------------- #
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.head_output(self.head_inter(x.view(x.size(0), -1)))
