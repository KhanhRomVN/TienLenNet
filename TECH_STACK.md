# 🛠 Technology Stack

## 📚 Core Libraries

| Library | Version | Purpose                 |
| ------- | ------- | ----------------------- |
| PyTorch | 2.0+    | Deep Learning Framework |
| NumPy   | 1.22+   | Scientific Computing    |
| tqdm    | 4.65+   | Progress Monitoring     |
| lz4     | 4.0+    | Data Compression        |
| pyarrow | 14.0+   | Efficient Data Storage  |

## 🖥️ Hardware Requirements

| Component | Recommended         |
| --------- | ------------------- |
| CPU       | 4+ cores (Intel i7) |
| RAM       | 16GB+               |
| GPU       | NVIDIA RTX 3060+    |
| Storage   | SSD 256GB+          |

## 🧩 System Architecture

```mermaid
graph LR
    A[Game Environment] --> B(MCTS)
    B --> C[Neural Network]
    C --> D[PPO Training]
    D --> E[Model Update]
    E --> A
```

## 🧪 Key Features

- **State Representation**: Tensor 6x4x13
- **Action Space**: 200 possible moves
- **Training**: PPO with Generalized Advantage Estimation
- **Efficiency**: LZ4 compression for experience replay
