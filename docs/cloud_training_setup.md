# Google Colab Training Notebook Setup for SmolVLA Fine-Tuning

This guide helps run SmolVLA fine-tuning on a free/cheap Cloud GPU (e.g. Google Colab T4/A100 or RunPod).

### 1. Clone repository & install dependencies

```bash
!git clone <your-repo-url>
%cd vla-manipulation
!pip install lerobot maniskin torch>=2.3.0
```

### 2. Run fine-tuning command

```bash
!lerobot-train --config training/finetune_config.yaml
```
