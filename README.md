# CartPole RL Project

Bu proje, DQN (Deep Q-Network) kullanarak CartPole-v1 ortamında bir ajan eğitmek için geliştirilmiştir.

# CartPole Reinforcement Learning Project

This project demonstrates how to train a reinforcement learning (RL) agent to solve the classic **CartPole** problem using PyTorch and GPU acceleration (CUDA). The agent is trained using a custom neural network and is evaluated based on its performance in balancing the CartPole.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Quick Test](#quick-test)
  - [Full Training](#full-training)
  - [Testing the Model](#testing-the-model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

### Prerequisites

Ensure you have the following installed on your system:
- Python 3.9 or higher
- CUDA-compatible GPU (e.g., NVIDIA GeForce RTX 4060) and the associated CUDA libraries
- Virtual environment support (optional but recommended)

### Steps

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/cartpole-rl-cuda.git
   cd cartpole-rl-cuda
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Verify GPU availability:
   ```bash
   python
   >>> import torch
   >>> print(torch.cuda.is_available())
   True
   ```

---

## Usage

### Quick Test

Run a quick test for training using fewer episodes to verify the setup:
```bash
python src/train.py --episodes 100 --batch_size 32
```

### Full Training

Train the RL model on the CartPole environment:
```bash
python src/train.py --episodes 1000 --batch_size 64
```

### Visualize Training Metrics

After training, metrics such as episode rewards, lengths, and loss values are visualized using `matplotlib`. Check the generated graphs in the `notebook.ipynb` file or use the `cartpole_rl_cuda_results.pdf` for a summary.

### Testing the Model

Evaluate the trained model:
```bash
python src/test.py
```
Example output:
```
Test Episode 1 Total Reward: 500.0
Test Episode 2 Total Reward: 500.0
...
Test Ortalama Ödül: 500.00
Test En Yüksek Ödül: 500.0
```

## Results

### Training Performance

- **Goal**: The agent’s performance target is to achieve an average reward of 195 over 100 consecutive episodes.
- **Metrics**:
  - **Best Reward**: `500.0`
  - **Average Reward (Last 100 Episodes)**: `243.41`

### Visualizations
Training metrics (`episode_rewards`, `episode_lengths`, `losses`) are saved in `data/training_metrics.json`. You can visualize these using `notebook.ipynb`.


