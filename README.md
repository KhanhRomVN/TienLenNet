# 🃏 Tiến Lên Reinforcement Learning Project

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Dự án áp dụng Reinforcement Learning (RL) để huấn luyện AI chơi game Tiến Lên Miền Nam. Agent được train bằng Deep Q-Learning (DQN) với cơ chế reward đặc biệt.

## 🚀 Cách Chạy Chương Trình

```bash
git clone https://github.com/KhanhRomVN/tien_len_rl.git
cd tien_len_rl
pip install -r requirements.txt

# Train model mới
python train.py --episodes 10000 --batch_size 64

# Đánh giá AI
python evaluate.py --model_path models/best_model.pth
```

````

## 📂 Cấu Trúc Repo

```
tien_len_rl/
├── agents/              # Thuật toán RL
├── environment/         # Game engine Tiến Lên
├── models/              # Model đã train
├── utils/               # Tiện ích hỗ trợ
├── train.py             # Script training
├── evaluate.py          # Đánh giá hiệu suất
└── requirements.txt     # Thư viện cần thiết
```

## 📊 Kết Quả Ban Đầu

| Metric          | Value (5000 episodes) |
| --------------- | --------------------- |
| Win Rate        | 68.3%                 |
| Avg. Turns      | 12.4                  |
| High Card Usage | Optimized             |

## 📧 Liên Hệ

**KhanhRomVN**
[![Email](https://img.shields.io/badge/Gmail-khanhromvn%40gmail.com-red)](mailto:khanhromvn@gmail.com)

````

python3 -m src.game.main
python src/ai/test_agent_actions.py
