import numpy as np

class ReplayBuffer:
    def __init__(self, state_dim, max_size=10000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # Pre allocation
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, 1), dtype=np.int64) 
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        # True 1 False 0
        self.done = np.zeros((max_size, 1), dtype=np.float32) 


    def add(self, state, action, reward, next_state, done):

        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):


        batch_size = min(batch_size, self.size)
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.state[indices],
            self.action[indices],
            self.reward[indices],
            self.next_state[indices],
            self.done[indices]
        )