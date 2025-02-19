import random
import numpy as np

class ReplayBuffer:
    """Replay Buffer sınıfı"""
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def add(self, transition):
        """Transition ekle: (state, action, reward, next_state, done)"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Rastgele bir batch al"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)

class TrainingMetrics:
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
    
    def add_episode(self, reward, length):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
    
    def add_loss(self, loss):
        self.losses.append(loss)