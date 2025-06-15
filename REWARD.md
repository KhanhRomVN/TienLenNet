````markdown
# 🏆 Reward System Design

## 🏁 Terminal Rewards

| Result     | Value | Condition              |
| ---------- | ----- | ---------------------- |
| Win        | +100  | Hết bài đầu tiên       |
| 2nd Place  | +20   | Hết bài thứ hai        |
| 3rd Place  | -10   | Còn ít bài hơn người 4 |
| Last Place | -30   | Còn nhiều bài nhất     |

## ⚡ Immediate Rewards

| Action               | Value     | Example Case                 |
| -------------------- | --------- | ---------------------------- |
| Play Valid Cards     | +0.1/card | Đánh đôi 8 ➔ +0.2            |
| Successfully Block   | +1.0      | Chặt đôi 10 bằng đôi J       |
| Force Opponents Pass | +0.5      | Đánh tứ quý cả bàn bỏ lượt   |
| Get Blocked          | -0.8      | Đánh đôi 3 bị chặt           |
| Illegal Move         | -1.0      | Đánh 3♠ khi chưa hết vòng 2♦ |

## 🧠 Strategic Rewards

| Strategy              | Value | Trigger Condition       |
| --------------------- | ----- | ----------------------- |
| Hold High Cards       | +0.3  | Giữ 2/A/K đến 5 lá cuối |
| Reduce 50% Cards      | +5.0  | Khi bài còn ≤ 6 lá      |
| Fewest Cards in Round | +0.2  | Có ít bài nhất vòng     |
| Waste High Card Early | -0.5  | Đánh 2 trong 5 nước đầu |

## 🧮 Reward Calculation Pseudocode

```python
def calculate_reward():
    total = 0

    if game_over:
        total += get_terminal_reward(rank)

    if action == 'play':
        total += 0.1 * num_cards
        total += 1.0 if blocked_opponent else 0
        total -= 0.8 if got_blocked else 0

    if hold_high_cards and remaining_cards < 5:
        total += 0.3

    if initial_cards * 0.5 >= current_cards:
        total += 5.0

    return total
```
````

```

```
