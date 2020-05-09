import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_size = 128, fc2_size = 64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        # floor of (n + 2p -f)  / s + 1
        self.state_size = state_size

        self.conv1 = nn.Conv2d(3, 32, 3, 2)      # 41 * 41 * 32
        self.conv2 = nn.Conv2d(32, 64, 5)     # 37 * 37 * 64
        self.fc1 = nn.Linear(87616, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, action_size)

    def output_size(n, p, f, s):
        output_size = int ((n + 2*p - f) / s) + 1

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        # x = x.view(- 1, 87616) 
        x = x.flatten(1)   
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


