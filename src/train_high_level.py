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

# with this config we can run multiple experiments as instructed
RUN_NAME = "run3"  # "baseline", "run2", "run3"

# Output directory for plots (docs/images for GitHub Pages submission)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, "..", "docs", "images")
os.makedirs(PLOT_DIR, exist_ok=True)

#  CHANGED HYPERPARAMETERS FROM  AS INSTRUCTOD
N_ACTIONS = 8
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10000
LEARNING_RATE = 0.0001
BATCH_SIZE = 256
TAU = 0.005
UPDATE_FREQ = 4 
BUFFER_LENGTH = 10000
EPISODES = 2500


# CUDA if avaible else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



#  SETUP 
env = Hw2Env(n_actions=N_ACTIONS, render_mode="offscreen")

#  reseting the env and extract it to see how big it is.
env.reset()

initial_state = env.high_level_state()
state_dim = len(initial_state)


replay_buffer = ReplayBuffer(state_dim, max_size=BUFFER_LENGTH)

main_network = DQN_MLP(state_dim, N_ACTIONS).to(device)
target_network = DQN_MLP(state_dim, N_ACTIONS).to(device)

# Synchronizeing the target network for step 0 so they start identical.


target_network.load_state_dict(main_network.state_dict())

optimizer = optim.Adam(main_network.parameters(), lr=LEARNING_RATE) #Adam optimizer as always

global_step = 0
EPSILON = EPS_START

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

        #  only train if have enough data in the buffer AND it is time to update
        if replay_buffer.size >= BATCH_SIZE and global_step % UPDATE_FREQ == 0:
            batch = replay_buffer.sample(BATCH_SIZE)
         
            
            loss = compute_loss(main_network, target_network, batch, GAMMA, device)
            optimizer.zero_grad() #clear old gradients
            loss.backward() #compute new gradients
            optimizer.step() #update the weights
            
     #happens every time the main network updates
            update_target_network(main_network, target_network, TAU)
            
        # EPSILON DECAY: Now calculated continuously based on global_step
        EPSILON = update_epsilon(global_step, EPS_START, EPS_END, EPS_DECAY)
            

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
    
    
    if episode % 10 == 0 or episode == EPISODES - 1:
        print(f"Episode: {episode}, Steps: {episode_steps}, Reward: {cumulative_reward:.2f}, RPS: {rps:.2f}, Epsilon: {EPSILON:.3f}")



print("Training Completed, saving model and generating plots....")

# Save the trained model weights
model_path = f"dqn_high_level_{RUN_NAME}.pth"
torch.save(main_network.state_dict(), model_path)
print(f"Model saved to '{model_path}'")

# Plot 1 Total Reward per Episode
plt.figure(figsize=(10, 5))
plt.plot(reward_history, label="Total Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Training: Total Cumulative Reward per Episode")
plt.legend()
reward_plot_path = os.path.join(PLOT_DIR, f"{RUN_NAME}_reward.png")
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
rps_plot_path = os.path.join(PLOT_DIR, f"{RUN_NAME}_rps.png")
plt.savefig(rps_plot_path)
plt.close()
print(f"RPS plot saved to '{rps_plot_path}'")
