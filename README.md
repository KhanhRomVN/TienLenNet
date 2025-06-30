# 🃏 Tiến Lên AI với AlphaZero Approach

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Dự án áp dụng Reinforcement Learning (PPO + MCTS) để huấn luyện AI chơi game Tiến Lên Miền Nam.

## 🚀 Cách Chạy Chương Trình

```bash
git clone https://github.com/KhanhRomVN/tien_len_rl.git
cd tien_len_rl

# Cài đặt thư viện
pip install torch numpy tqdm lz4 pyarrow

# Train model
python tienlen_net_v1.py
```

## 📂 Cấu Trúc Code Chính

```python
# tienlen_net_v1.py
├── TienLenGame            # Game engine
├── TienLenNet             # Neural Network
├── ResidualBlock          # Khối residual CNN
├── Node                   # Node MCTS
├── MCTS                   # Monte Carlo Tree Search
├── PPOBuffer              # Experience Replay
├── ReplayBuffer           # Lưu trữ trajectory
├── ppo_train              # Huấn luyện PPO
├── self_play_game         # Tự chơi và thu thập dữ liệu
└── main_train_loop        # Vòng lặp huấn luyện chính
```

## 📊 Kết Quả

Sau 500 episodes huấn luyện:

- Win rate: ~60% (vs 3 bot rule-based)
- Thời gian huấn luyện: ~12h trên GPU

## 📧 Liên Hệ

**KhanhRomVN**  
[![Email](https://img.shields.io/badge/Gmail-khanhromvn%40gmail.com-red)](mailto:khanhromvn@gmail.com)
