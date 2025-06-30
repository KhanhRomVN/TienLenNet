# 🎓 Building Tiến Lên AI: Step-by-Step

## 1. Biểu diễn trạng thái (State Representation)

```python
def get_state(self):
    """Tensor 6x4x13 biểu diễn:
    0: Bài trên tay
    4: Combo hiện tại
    5: Lịch sử nước đi"""
    state = np.zeros((6, 4, 13), dtype=np.float32)
    # ... mã hóa bài ...
    return state
```

## 2. Xác định hành động hợp lệ

```python
def get_valid_combos(self, player_idx):
    valid_combos = []
    # PASS luôn hợp lệ
    valid_combos.append(("PASS", []))
    # Singles
    valid_combos.extend([("SINGLE", [card]) for card in player.hand])
    # Pairs
    for rank, cards in rank_counts.items():
        if len(cards) >= 2:
            valid_combos.append(("PAIR", cards[:2]))
    # ... các combo khác ...
    return valid_combos
```

## 3. Kiến trúc mạng Neural

```python
class TienLenNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(6, 128, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(...)

        # Policy head
        self.policy_conv = nn.Conv2d(128, 4, kernel_size=1)
        self.policy_out = nn.Linear(256, 200)

        # Value head
        self.value_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.value_out = nn.Linear(128, 1)
```

## 4. MCTS với Continual Resolving

```python
class MCTS:
    def run(self, game):
        root = Node(game.copy(), model=self.model)
        for _ in range(self.num_simulations):
            node = root
            # Selection
            while not node.is_leaf():
                node = node.select_child()
            # Expansion
            if not node.resolved:
                node.resolve()
            # Simulation
            value = node.resolve() if not node.game_instance.done else reward
            # Backpropagation
            node.update(value)
```

## 5. Huấn luyện với PPO

```python
def ppo_train(model, buffer, optimizer, scaler, clip_epsilon=0.2, ...):
    # Compute advantages
    advantages = buffer.compute_advantages()

    # PPO updates
    for _ in range(ppo_epochs):
        # Policy loss
        ratio = (new_log_probs - old_log_probs).exp()
        policy_loss = -torch.min(ratio * advantages,
                                torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon) * advantages).mean()

        # Update model
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
```

## 6. Tối ưu hiệu năng

- **State Caching**: Lưu trữ valid actions
- **Vectorization**: Mã hóa bài dạng tensor
- **Compression**: Nén dữ liệu replay với LZ4

```python
class ReplayBuffer:
    def push(self, trajectory):
        compressed = lz4.frame.compress(pickle.dumps(trajectory))
        self.buffer.append(compressed)
```

## 7. Chạy huấn luyện

```python
if __name__ == "__main__":
    # GPU setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = TienLenNet().to(device)

    # Start training
    main_train_loop()
```
