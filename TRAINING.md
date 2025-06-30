# 🚂 Training Procedure

## ⚙️ Hyperparameters

```python
# Trong main_train_loop()
num_episodes = 500
games_per_episode = 30
training_steps = 100
min_buffer_size = 500
batch_size = 256
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
```

## 📉 Loss Functions

1. **Policy Loss**:

```python
policy_loss = -torch.min(surr1, surr2).mean()
```

2. **Value Loss**:

```python
value_loss1 = F.mse_loss(value_pred, returns)
value_loss2 = F.mse_loss(value_pred_clipped, returns)
value_loss = torch.max(value_loss1, value_loss2).mean()
```

3. **Entropy Bonus**:

```python
entropy_loss = -dist.entropy().mean()
```

## 📊 Performance Tracking

| Episode Range | Win Rate | Avg. Reward |
| ------------- | -------- | ----------- |
| 1-100         | 15-25%   | -2.1        |
| 101-300       | 40-50%   | +5.8        |
| 301-500       | 55-65%   | +12.4       |

## 💾 Model Checkpoints

```bash
saved_models/
├── tien_len_net_ep100.pth
├── tien_len_net_ep300.pth
├── tien_len_net_ep500.pth
└── best_tien_len_net_ep420_win0.65.pth
```

## 🧪 Evaluation Protocol

```python
def evaluate(model, num_games=20):
    model_wins = 0
    for game_id in range(num_games):
        # ... play full game ...
        if game.winner == 0:
            model_wins += 1
    return model_wins / num_games
```
