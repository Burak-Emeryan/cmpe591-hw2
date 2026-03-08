import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from homework2 import Hw2Env
from replay_buffer import ReplayBuffer
from networks import DQN_MLP
from utils import select_action, compute_loss, update_target_network, update_epsilon

# Output directory for plots (docs/images for GitHub Pages submission)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, "..", "docs", "images")
os.makedirs(PLOT_DIR, exist_ok=True)

#  HYPERPARAMETERS 
N_ACTIONS = 8
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.999 
EPSILON_DECAY_ITER = 10 
MIN_EPSILON = 0.1 
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
UPDATE_FREQ = 4 
TARGET_NETWORK_UPDATE_FREQ = 100 
BUFFER_LENGTH = 10000
EPISODES = 500


# CUDA if avaible else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



#  SETUP 
env = Hw2Env(n_actions=N_ACTIONS, render_mode="gui")

#  reseting the env and extract it to see how big it is.
env.reset()

initial_state = env.high_level_state()
state_dim = len(initial_state)


replay_buffer = ReplayBuffer(state_dim, max_size=BUFFER_LENGTH)

main_network = DQN_MLP(state_dim, N_ACTIONS).to(device)
target_network = DQN_MLP(state_dim, N_ACTIONS).to(device)

# Synchronizeing the target network for step 0 so they start identical.


update_target_network(main_network, target_network)

optimizer = optim.Adam(main_network.parameters(), lr=LEARNING_RATE) #adam optimizer as always

global_step = 0

# History lists for plotting
reward_history = []
rps_history = []

#  THE GRAND LOOP 
for episode in range(EPISODES):
    env.reset()
    state = env.high_level_state()
    done = False
    cumulative_reward = 0.0
    episode_steps = 0
    
    while not done:
        
        action = select_action(main_network, state, EPSILON, N_ACTIONS, device)
      
        _, reward, is_terminal, is_truncated = env.step(action)
        next_state = env.high_level_state()
        done = is_terminal or is_truncated
        
        replay_buffer.add(state, action, reward, next_state, done)

        # TRAINING  

        # We only train if we have enough data in the buffer AND it is time to update
        if replay_buffer.size >= BATCH_SIZE and global_step % UPDATE_FREQ == 0:
            batch = replay_buffer.sample(BATCH_SIZE)
         
            
            loss = compute_loss(main_network, target_network, batch, GAMMA, device)
            optimizer.zero_grad() #clear old gradients
            loss.backward() #compute new gradients
            optimizer.step() #update the weights
            
        # EPSILON DECAY 
        if global_step % EPSILON_DECAY_ITER == 0:
    
            EPSILON = update_epsilon(EPSILON, MIN_EPSILON, EPSILON_DECAY)
            
        #  TARGET NETWORK SYNC 
        if global_step % TARGET_NETWORK_UPDATE_FREQ == 0:
            update_target_network(main_network, target_network)
        # next state
        state = next_state
        cumulative_reward += reward
        episode_steps += 1
        global_step += 1
    

    # Calculate RPS
    rps = cumulative_reward / episode_steps
    
    # Store metrics for plotting
    reward_history.append(cumulative_reward)
    rps_history.append(rps)
    
    # Printing metrics at the end of every episode
    print(f"Episode: {episode}, Steps: {episode_steps}, Reward: {cumulative_reward:.2f}, RPS: {rps:.2f}, Epsilon: {EPSILON:.3f}")


print("Training Completed, saving model and generating plots....")

# Save the trained model weights
torch.save(main_network.state_dict(), "dqn_high_level_model.pth")
print("Model saved to 'dqn_high_level_model.pth'")

# Plot 1 Total Reward per Episode
plt.figure(figsize=(10, 5))
plt.plot(reward_history, label="Total Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Training: Total Cumulative Reward per Episode")
plt.legend()
reward_plot_path = os.path.join(PLOT_DIR, "hw2_reward_plot.png")
plt.savefig(reward_plot_path)
plt.close()
print(f"Reward plot saved to '{reward_plot_path}'")

# Plot 2 as  Reward Per Step perEpisode
plt.figure(figsize=(10, 5))
plt.plot(rps_history, label="Reward Per Step (RPS)", color="orange")
plt.xlabel("Episode")
plt.ylabel("RPS")
plt.title("DQN Training: Reward Per Step over Episodes")
plt.legend()
rps_plot_path = os.path.join(PLOT_DIR, "hw2_rps_plot.png")
plt.savefig(rps_plot_path)
plt.close()
print(f"RPS plot saved to '{rps_plot_path}'")
