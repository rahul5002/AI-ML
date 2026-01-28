import torch.nn as nn
import torch.optim as optim
import torch

# Define N_FEATURES based on feature extraction output
N_FEATURES = 100 
N_ACTIONS = 2 

# Step 2: Defining the Q-Network (Online and Target) 
class PhishingQNetwork(nn.Module):
    """Defines the neural network structure for Q-value approximation."""
    def __init__(self, input_size, output_size):
        super(PhishingQNetwork, self).__init__()
        # A 3-layer fully connected architecture suitable for structured feature vectors 
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size) 
        )

    def forward(self, x):
        return self.fc(x)

# Initialization of Online (Policy) and Target Networks
policy_net = PhishingQNetwork(N_FEATURES, N_ACTIONS)
target_net = PhishingQNetwork(N_FEATURES, N_ACTIONS)
# Copy initial weights and freeze target network for stability 
target_net.load_state_dict(policy_net.state_dict())
target_net.eval() 

# Optimizer Definition
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)