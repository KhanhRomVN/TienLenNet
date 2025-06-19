import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    """Mạng nơ-ron sâu cho DQN"""
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),   # Smaller layer
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        self.half()  # Use float16 for memory saving
 
    def forward(self, x):
        # For fp16 support: type cast if needed
        if x.dtype != torch.float16:
            x = x.half()
        return self.fc(x)

class ReplayBuffer:
    """Bộ nhớ đệm cho trải nghiệm"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.policy_net = DQN(state_size, action_size).to(self.device).half()
        self.target_net = DQN(state_size, action_size).to(self.device).half()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(10000)
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        """Chọn hành động dựa trên epsilon-greedy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device).half()
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_model(self, batch_size):
        if len(self.memory) < batch_size:
            return

        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device).half()
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device).half()
        next_states = torch.FloatTensor(next_states).to(self.device).half()
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device).half()

        # Tính Q-values hiện tại
        current_q = self.policy_net(states).gather(1, actions)

        # Tính Q-values mục tiêu
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Cập nhật mô hình
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        loaded_state = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(loaded_state)
        self.target_net.load_state_dict(self.policy_net.state_dict())
# Patch: enable running agent.py or model.py standalone 
if __name__ == "__main__":
    from src.ai.model import DQNAgent
    # Allow manual test run for model structure
    print("DQNAgent:", DQNAgent)