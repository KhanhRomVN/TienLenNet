````markdown
# 🚂 Training Procedure

## 📦 Dataset Specifications

- 100,000+ game states
- 15-dimensional state vector:
  ```python
  [card_count, has_2, has_A, ..., current_suit, turn_number]
  ```
````

- Action space: 150 possible moves

## ⚙️ Hyperparameters

```yaml
episodes: 10000
batch_size: 64
gamma: 0.95
epsilon:
  start: 1.0
  end: 0.01
  decay: 0.9995
learning_rate: 0.00025
target_update: 1000
```

## 📉 Loss Convergence

![Training Loss Curve](docs/images/loss_curve.png)

## 📊 Performance Metrics

| Episode Range | Win Rate | Avg. Reward |
| ------------- | -------- | ----------- |
| 1-1000        | 12.3%    | -5.2        |
| 1001-5000     | 41.7%    | +8.6        |
| 5001-10000    | 68.3%    | +24.1       |

## 💾 Model Checkpoints

```bash
models/
├── episode_5000.pth
├── episode_8000.pth
└── best_model.pth   # Highest win rate
```

## 🧪 Evaluation Protocol

```python
def evaluate(model, num_episodes=100):
    wins = 0
    for _ in range(num_episodes):
        if play_game(model) == 1:
            wins += 1
    return wins / num_episodes
```

```

```
