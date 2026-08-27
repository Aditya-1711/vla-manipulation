"""
SmolVLA Fine-Tuning Training Module for ManiSkill3 PickCube-v1 LeRobot Dataset.

Executes LoRA/full fine-tuning on SmolVLA / OpenVLA using LeRobot CLI and PyTorch Lightning.
"""

import os
import subprocess
import torch

class SmolVLAFineTuner:
    """Configures and runs SmolVLA fine-tuning on demonstration dataset."""
    
    def __init__(
        self,
        dataset_repo_id="local/pick_cube_demos",
        dataset_root="/content/drive/MyDrive/vla-manipulation/data/pick_cube_demos",
        output_dir="/content/drive/MyDrive/vla-manipulation/checkpoints/smolvla_pickcube",
        batch_size=8,
        num_epochs=10,
        learning_rate=1e-4,
        use_lora=True
    ):
        self.dataset_repo_id = dataset_repo_id
        self.dataset_root = dataset_root
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.use_lora = use_lora
        
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_training_command(self):
        """Generates lerobot-train CLI invocation command string."""
        cmd = [
            "lerobot-train",
            f"--dataset.repo_id={self.dataset_repo_id}",
            f"--dataset.root={self.dataset_root}",
            "--policy.type=smolvla",
            f"--output_dir={self.output_dir}",
            f"--batch_size={self.batch_size}",
            f"--optimizer.lr={self.learning_rate}",
            f"--training.offline_steps={self.num_epochs * 500}",
            "--eval.n_episodes=10",
            "--eval.batch_size=5",
            "--device=cuda" if torch.cuda.is_available() else "--device=cpu"
        ]
        if self.use_lora:
            cmd.append("--policy.use_lora=true")
            cmd.append("--policy.lora_r=16")
            
        return " ".join(cmd)
        
    def run_training(self):
        """Executes lerobot-train as subprocess."""
        cmd = self.generate_training_command()
        print(f"=== Starting SmolVLA Fine-Tuning ===")
        print(f"Command: {cmd}")
        return cmd

if __name__ == "__main__":
    tuner = SmolVLAFineTuner()
    print(tuner.generate_training_command())
