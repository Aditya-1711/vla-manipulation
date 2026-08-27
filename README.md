# VLA Fine-Tuning for Robotic Manipulation

Fine-tuning an open-source Vision-Language-Action (VLA) model (**SmolVLA**, 450M parameters) on a pick-and-place task in simulation (**ManiSkill3** / **LeRobot**), comparing performance and generalization against a classical IK/motion-planning baseline.

## Project Structure

```
.
├── README.md                  # Problem statement, results summary, demo clip
├── sim_env/
│   ├── task_def.py            # ManiSkill3 pick-place task + randomization config
│   └── scripted_baseline.py   # IK/motion-planning baseline policy
├── data_collection/
│   ├── collect_demos.py       # Scripted or teleoperated demo recording → LeRobot dataset format
│   └── dataset_stats.py       # Sanity checks: episode count per variation, replay validation
├── training/
│   ├── finetune_config.yaml   # lerobot-train config (policy, dataset, steps, batch size)
│   └── train.sh               # Wraps lerobot-train invocation
├── evaluation/
│   ├── eval_suite.py          # Runs both policies across in-distribution + generalization trials
│   ├── metrics.py             # Success rate, grasp success, completion time, failure tags
│   └── results/                # Raw logs, CSVs, plots
├── media/
│   └── demo_comparison.mp4    # Side-by-side rollout clip
└── docs/
    └── writeup.md             # Final report: method, results, failure analysis
```

## Phased Workflow Progress

- [x] **Phase 0**: Repo Scaffolding & Setup
- [ ] **Phase 1**: Task Definition & Scripted Baseline
- [ ] **Phase 2**: Demonstration Data Collection & Replay Validation
- [ ] **Phase 3**: SmolVLA Fine-Tuning Pipeline
- [ ] **Phase 4**: In-Distribution & Generalization Evaluation Suite
- [ ] **Phase 5**: Results Summary & Side-by-Side Demo
