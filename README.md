# SmolVLA Robotic Manipulation Pipeline (ManiSkill3 + LeRobot)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)

An end-to-end Vision-Language-Action (VLA) manipulation pipeline fine-tuning **SmolVLA** (500M parameters) on **ManiSkill3 PickCube-v1** using **HuggingFace LeRobot**.

---

## 📌 Features

- **Phase 0 & 1: Baseline Simulation**: Scripted pick-and-place policy achieving **95% In-Distribution** and **100% Generalization** success on ManiSkill3 Franka `PickCube-v1`.
- **Phase 2: Demonstration Collection**: Rebalanced 60-episode dataset containing 2,988 frames with 7D TCP state (`observation.state`) and RGB observations formatted for LeRobot.
- **Phase 3: VLA Fine-Tuning**: Integration with `lerobot/smolvla_base` fine-tuning pipeline, logging loss progression from 2.092 to 0.264 (~87.4% reduction).
- **Phase 4: Decoupled Rollout & Evaluation**: Official 3-step preprocessor $\to$ policy $\to$ postprocessor inference pipeline evaluating 35 benchmark seeds across 4 categories:
  - Seen Training Seeds (10 seeds)
  - Verified In-Distribution Unseen Seeds (5 seeds)
  - Verified Generalization Unseen Seeds (10 seeds)
  - Fresh Held-Out Unseen Seeds (10 seeds)

---

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/Aditya-1711/vla-manipulation.git
cd vla-manipulation

# Install core dependencies
pip install "numpy>=2.0.0,<2.3.0"
pip install "lerobot[dataset]" av wandb num2words
pip install --no-deps mani_skill
pip install pytorch_kinematics_ms arm_pytorch_utilities nvidia-ml-py
pip install gymnasium h5py trimesh transforms3d pandas sapien dacite GitPython tyro
```

---

## 🚀 Usage

### 1. Fine-Tuning SmolVLA Policy
```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/pick_cube_rebalanced \
  --dataset.root="./data/lerobot_pick_cube" \
  '--rename_map={"observation.image": "observation.images.camera1"}' \
  --output_dir="./checkpoints/smolvla_pickcube_60ep_8k" \
  --batch_size=4 \
  --optimizer.lr=1e-4 \
  --steps=8000 \
  --save_freq=1000 \
  --log_freq=20 \
  --wandb.enable=false
```

### 2. Single Episode Diagnostic Rollout
```bash
python evaluation/eval_seed1000_full_trace.py
```

### 3. Complete 35-Seed Benchmark Evaluation
```bash
python evaluation/eval_full_matrix.py
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
