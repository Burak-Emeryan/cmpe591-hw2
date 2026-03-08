import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN_MLP(nn.Module):
    def __init__(self, state_dim, n_actions):
        super(DQN_MLP, self).__init__()
        
        self.layer1 = nn.Linear(state_dim,128) #input layer to first hiden layer
        self.layer2 = nn.Linear(128,128) # first hidden layer to second hidden layer
        self.layer3 = nn.Linear(128,n_actions) # second hidden layer to output layer

    def forward(self, x):
       #linear transformation then ReLu activation 
        x = F.relu(self.layer1(x))                                          
        x = F.relu(self.layer2(x))
        x = self.layer3(x) #ReLu did not applied for possible negative Q values as penalty
        return x