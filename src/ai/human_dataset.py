import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

class HumanDemonstrationDataset(Dataset):
    """
    Dataset chứa trạng thái + hành động (action_id) lấy từ log người chơi thật
    game_records: List[Dict] với key 'state', 'action_id'
    """
    def __init__(self, game_records, encode_state_func):
        self.states = []
        self.actions = []
        for record in game_records:
            self.states.append(encode_state_func(record['state']))
            self.actions.append(record['action_id'])

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

def pretrain_with_human_data(agent, dataset, device, epochs=50):
    """
    Pretrain NN của agent theo dữ liệu mẫu (Supervised Learning)!
    agent: đối tượng Agent với agent.policy_net kiểu nn.Module, đã sẵn sàng huấn luyện.
    dataset: HumanDemonstrationDataset
    device: torch.device()
    """
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    net = agent.agent.policy_net
    net.train()
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for state, target_action in loader:
            state = state.to(device)
            target_action = target_action.to(device)

            q_values = net(state)
            loss = criterion(q_values, target_action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Pretrain epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")