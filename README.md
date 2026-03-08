# CMPE591 – Deep Learning in Robotics
### Homework 2: Deep Q-Network (DQN)

This repo is   for HW2 of CMPE591. The goal is to train a DQN agent that learns to push a box to a target position in a MuJoCo simulation using a robotic arm.

---


## What this homework is about

The agent controls a robot arm that moves in 8 discrete directions. At each step it picks an action, the arm moves, and it gets a reward based on how close the end-effector is to the box and how close the box is to the green goal marker. Episode ends when the box reaches the goal or after 50 steps whichever comes first.

Reward function:
```
reward = 1/distance(end-effector, object) + 1/distance(object, goal)
```

---

##  Aproach

I used `high_level_state()` instead of raw pixel observations. The high-level state is a 6-dimensional vector:
- `[ee_x, ee_y, obj_x, obj_y, goal_x, goal_y]`

This lets me use a simple MLP instead of a convolutional network, which is much faster to train. The instructor explicitly mentioned this as a valid approach.

---

## File structure

```
src/
├── train_high_level.py   # main training script
├── networks.py           # MLP network definition (DQN_MLP)
├── utils.py              # select_action, compute_loss, update_target, update_epsilon
├── replay_buffer.py      # experience replay buffer
├── homework2.py          # environment (provided by instructor)
└── environment.py        # base environment (provided by instructor)

docs/
├── homeworks.html        # GitHub Pages submission site
└── images/
    ├── hw2_reward_plot.png   # reward over episodes plot
    └── hw2_rps_plot.png      # reward per step over episodes plot
```

---

## Network architecture

Simple 3-layer MLP (since we use high-level state, no CNN needed for this job):

```
Input(6) → Linear(128) → ReLU → Linear(128) → ReLU → Linear(8)
```

Output is Q-values for each of the 8 actions. No ReLU on the last layer since Q-values can be negative.

---

## Hyperparameters

```python
N_ACTIONS      = 8
GAMMA          = 0.99
EPSILON        = 1.0       # starts fully random
EPSILON_DECAY  = 0.999     # decay factor applied every 10 steps
EPSILON_MIN    = 0.1       # never goes below 10% random
LEARNING_RATE  = 0.0001
BATCH_SIZE     = 32
UPDATE_FREQ    = 4         # train every 4 steps
TARGET_UPDATE  = 100       # copy weights to target network every 100 steps
BUFFER_LENGTH  = 10000
EPISODES       = 500
```

These match the suggested hyperparameters from the homework instructions.

---

## How to run

**1. Set up the environment (CPython 3.9 required — PyTorch does not support PyPy):**
```bash
conda create -n cmpe591 python=3.9 -y
conda activate cmpe591
```

**2. Install dependencies:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install dm_control==1.0.10 mujoco==2.3.2 matplotlib numpy pyyaml
pip install git+https://github.com/alper111/mujoco-python-viewer.git
```

**3. Run training:**
```bash
cd src
python train_high_level.py
```

After 500 episodes, the script saves:
- `dqn_high_level_model.pth` — trained model weights
- `docs/images/hw2_reward_plot.png` — cumulative reward per episode
- `docs/images/hw2_rps_plot.png` — reward per step per episode

---

## Training details

- The replay buffer starts filling up from episode 1. Training only kicks in once there are at least 32 transitions in the buffer.
- Epsilon decays from 1.0 to 0.1 over the course of training, transitioning from pure exploration to mostly greedy behavior.
- The target network is a frozen copy of the main network that gets hard-updated every 100 steps. This stabilizes training by keeping the Q-targets fixed for a while.
- Adam optimizer with lr=0.0001, MSE loss on Bellman targets.

---

## Results

Training curves are embedded in the submission page at `docs/homeworks.html`.

The reward and RPS both trend upward over 500 episodes and this shows that  the agent learns to navigate the arm toward the box and push it closer to the goal over time.
