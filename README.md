# CMPE591 – Deep Learning in Robotics
### Homework 2: Deep Q-Network (DQN)

This repo is   for HW2 of CMPE591. The goal is to train a DQN agent that learns to push a box to a target position in a MuJoCo simulation using a robotic arm.


## What this homework is about

The agent controls a robot arm that moves in 8 discrete directions. At each step it picks an action, the arm moves, and it gets a reward based on how close the end-effector is to the box and how close the box is to the green goal marker. Episode ends when the box reaches the goal or after 50 steps whichever comes first.

Reward function expresed as:
```
reward = 1/distance(end-effector, object) + 1/distance(object, goal)
```

---

##  Aproach

I used `high_level_state()` instead of raw pixel observations as advides in the isntructions. The high level state is a 6 dimensional vector as :
- `[ee_x, ee_y, obj_x, obj_y, goal_x, goal_y]`

This lets  usage of a simple MLP instead of a convolutional network, which is much faster to train. The instructor explicitly mentioned this as a valid approach.

---



## Network architecture

Simple 3-layer MLP (since we use high-level state, no CNN needed for this job):

```
Input(6) → Linear(128) → ReLU → Linear(128) → ReLU → Linear(8)
```

Output is Q-values for each of the 8 actions. No ReLU on the last layer since Q-values can be negative.

---

## Modified  Hyperparameters & Code Updates after the new instructions

Based on the latest instructions, I updated the code as :
- **Episodes:** Increased to 2500.
- **Soft Updates:** Used an Exponential Moving Average (EMA) with `TAU = 0.005` to smoothly update the target network at every step.
- **Epsilon Decay:** Changed to a continuous exponential decay function.
- **Rendering:** Turned off GUI (`render_mode="offscreen"`) to speed up training.

```python
N_ACTIONS      = 8
GAMMA          = 0.99
EPS_START      = 0.9
EPS_END        = 0.05
EPS_DECAY      = 10000
LEARNING_RATE  = 0.0001
BATCH_SIZE     = 128
TAU            = 0.005
UPDATE_FREQ    = 4
BUFFER_LENGTH  = 10000
EPISODES       = 2500
```

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

After 2500 episodes, the script saves:
- `dqn_high_level_{RUN_NAME}.pth` — trained model weights
- `docs/images/{RUN_NAME}_reward.png` — cumulative reward per episode
- `docs/images/{RUN_NAME}_rps.png` — reward per step per episode

---

## Training details

- The replay buffer starts filling up from episode 1. Training only kicks in once there are enough transitions in the buffer.
- Epsilon decays from 0.9 to 0.05 over the course of training, transitioning from exploration to mostly greedy behavior.
- The target network is updated via soft updates (EMA) with TAU=0.005 at every training step. This stabilizes training by slowly blending new weights into the target.
- Adam optimizer with MSE loss on bellman targets.

---


## Hyperparameter Study & Performance Review

I ran the training 3 different runs to see how changing parameters effects the learning process. The generated plots are saved in the `docs/images/` folder.

### Run 1: The Baseline Model
**Settings:** `LEARNING_RATE = 0.0001`, `BATCH_SIZE = 128`

**Performance:** The agent learned successfully. Rewards went up steadily over time and spiked around 50 near the end as expected.



### Run 2: Increased Learning Rate
**Settings:** `LEARNING_RATE = 0.0005`, `BATCH_SIZE = 128`

**Performance:** The agent got higher rewards faster in the early episodes. However, the final plots were much noisier and more unstable compared to the baseline model.

**Main Reason:** A higher learning rate makes the optimizer take larger steps. It learns fast at first, but it overshoots the best values later on, causing unstable and erratic behavior.

### Run 3: Increased Batch Size
**Settings:** `LEARNING_RATE = 0.0001`, `BATCH_SIZE = 256`

**Performance:** The reward plots showed a thicker more consistent block of high rewards at the end. There were fewer sudden drops in performance.

**Main Reason:** A larger batch size means the network averages its updates over more past memories. This filters out random, bad steps and makes the final policy much more stable and reliable.

---

## Results

Training curves are embedded in the submission page at `docs/homeworks.html`.
