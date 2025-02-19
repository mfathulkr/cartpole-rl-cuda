import torch
import torch.nn as nn
import torch.optim as optim
import random

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4):
        # Device seçimi
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        ).to(self.device)  # GPU'ya taşı
        
        self.target_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        ).to(self.device)  # GPU'ya taşı
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        self.epsilon = 1.0  # Başlangıçta %100 keşif
        self.epsilon_min = 0.02  # Biraz daha yüksek
        self.epsilon_decay = 0.997  # Daha da yavaş decay

    def act(self, state):
        """Epsilon-greedy stratejisi ile aksiyon seç"""
        if random.random() < self.epsilon:
            return random.randint(0, self.policy_net[-1].out_features - 1)  # Keşif
        
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            q_values = self.policy_net(state)
            return torch.argmax(q_values).cpu().item()  # En iyi aksiyonu seç

    def train_step(self, replay_buffer, batch_size=64, gamma=0.99):
        if len(replay_buffer) < batch_size:
            return None
        
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        q_values = self.policy_net(states).gather(1, actions.view(-1, 1)).squeeze()
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
        target_q_values = rewards + gamma * next_q_values * (1 - dones)
        
        loss = self.criterion(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()

    def update_epsilon(self):
        """Epsilon değerini güncelle"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        """Target network'ü güncelle"""
        for target_param, param in zip(self.target_net.parameters(), 
                                     self.policy_net.parameters()):
            target_param.data.copy_(0.005 * param.data + 
                                  0.995 * target_param.data)  # Daha hızlı update

    def update_target_network_at_interval(self, episode):
        if episode % 5 == 0:  # Her 5 episode'da bir güncelle
            self.update_target_network()