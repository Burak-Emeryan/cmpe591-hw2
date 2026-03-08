import numpy as np
import torch
import torch.nn.functional as F

def select_action(network, state, epsilon, n_actions, device):

    if np.random.random() < epsilon: # Epsilongreedy policy
        return np.random.randint(n_actions) # Random action
    else:
        # Push the state tensor to the GPU
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device) 
        q_values = network(state_tensor) 
        # Push the Q values back to the CPU
        q_values = q_values.cpu() 
        return torch.argmax(q_values).item()
    
        


def compute_loss(main_network, target_network, batch, gamma, device):
   
  
    np_states, np_actions, np_rewards, np_next_states, np_dones = batch #unpacking Numpy arrays from the batch into individual components

    # Convert and push to GPU
    states = torch.FloatTensor(np_states).to(device) 
    actions = torch.LongTensor(np_actions).to(device)
    rewards = torch.FloatTensor(np_rewards).to(device)
    next_states = torch.FloatTensor(np_next_states).to(device)
    dones = torch.FloatTensor(np_dones).to(device)

    q_values = main_network(states) # Getting Q values from the main network
    
    current_q = q_values.gather(1, actions) # Gathering the Q values for the actions taken
    
    next_q_values = target_network(next_states) # Getting Q values from the target network

    max_next_q = next_q_values.max(1)[0].unsqueeze(1).detach() # Detaching the target network values with detach()

    y = rewards + (gamma * max_next_q* (1-dones)) # Bellman equation

    loss = F.mse_loss(current_q, y) # Mean Squared Error loss
    
    return loss

# Uptading the weights of the target network with the weights of the main network
def update_target_network(main_network, target_network): 
    target_network.load_state_dict(main_network.state_dict())

# Updating the epsilon value for the epsilon-greedy policy
def update_epsilon(epsilon, min_epsilon, decay_rate):
    return max(min_epsilon, epsilon * decay_rate)
