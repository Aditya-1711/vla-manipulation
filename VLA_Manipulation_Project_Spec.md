# Project Spec: VLA Fine-Tuning for Robotic Manipulation

**Owner:** Aditya Kapile
**Goal:** Fine-tune an open-source Vision-Language-Action (VLA) model on a pick-and-place task in simulation, and quantitatively demonstrate where a learned policy beats — and where it loses to — a classical scripted/planning baseline. Ship a repo, a results writeup, and a demo video.

This doc is written to be handed directly to an agentic coding tool (e.g. Antigravity) as a phased build plan. Each phase has a clear deliverable and exit criteria so the agent can self-verify before moving on.

---

## 1. Why this project (context, not for the agent)

Robotics/embodied-AI hiring in 2026 consistently asks for the same combination: VLA model work (vision-language-action policies), simulation-to-policy pipelines, and evaluation of generalization — not just a single demo. This project is designed to produce evidence of all three, plus a genuine comparison against classical robotics (which most VLA-only projects skip, and which is exactly where your existing ROS2/planning background adds credibility).

---

## 2. Success criteria

The project is "done" when you can say, truthfully:

- Fine-tuned SmolVLA (or a comparable open VLA) on a pick-and-place task, from a self-collected demonstration dataset.
- Built a scripted/classical baseline (motion planning or IK-based) for the same task.
- Ran both policies through an evaluation suite that includes **in-distribution** trials (matching training conditions) and **generalization** trials (novel object positions, novel object shapes/colors, distractor objects).
- Have hard numbers: success rate, grasp success rate, task completion time, failure mode breakdown, for both policies, across both eval conditions.
- Have a short demo video/GIF showing both policies attempting the same generalization trial side by side.

That last point matters most for interviews — a side-by-side clip where the VLA policy visibly handles an unseen object position and the scripted baseline fails (or vice versa) is a 30-second story that does more work than any bullet point.

---

## 3. Tech stack

| Component | Choice | Why |
|---|---|---|
| VLA model | **SmolVLA** (`lerobot/smolvla_base`, 450M params) | Small enough to fine-tune on a single consumer GPU (~10–16GB VRAM at batch size 8), actively maintained by Hugging Face, designed specifically for fine-tuning on custom LeRobot datasets. π0/OpenVLA are heavier and better as a "stretch" comparison if time allows. |
| Framework | **Hugging Face LeRobot** | Handles dataset format, training loop (`lerobot-train`), and policy rollout in one place — avoids building custom data plumbing. |
| Simulator | **ManiSkill3** | GPU-parallelized, SAPIEN-based, has procedural pick-place task generation with built-in domain randomization (object pose, geometry) — this is what makes the generalization test suite easy to build instead of hand-rolled. Alternative: **LIBERO**, the standard imitation-learning benchmark already used in the SmolVLA community if you want pre-existing task suites instead of building your own. |
| Baseline policy | Scripted pick-place using ManiSkill3's built-in IK/motion planning | Gives you the classical-robotics comparison point. |
| Tracking | Weights & Biases (`--wandb.enable=true` in `lerobot-train`) | Free tier is enough; gives you training curves for the writeup. |
| Compute | Cloud GPU (Colab A100, or rented RTX 4090/A100 instance) if no local CUDA GPU with ≥12GB VRAM | Local MPS (Mac) training is not practical for this — confirmed too slow in community reports. |

---

## 4. Repo structure

```
vla-manipulation/
├── README.md                  # problem statement, results summary, demo gif
├── sim_env/
│   ├── task_def.py            # ManiSkill3 pick-place task + randomization config
│   └── scripted_baseline.py   # IK/motion-planning baseline policy
├── data_collection/
│   ├── collect_demos.py       # scripted or teleoperated demo recording → LeRobot dataset format
│   └── dataset_stats.py       # sanity checks: episode count per variation, replay validation
├── training/
│   ├── finetune_config.yaml   # lerobot-train config (policy, dataset, steps, batch size)
│   └── train.sh               # wraps lerobot-train invocation
├── evaluation/
│   ├── eval_suite.py          # runs both policies across in-distribution + generalization trials
│   ├── metrics.py             # success rate, grasp success, completion time, failure categorization
│   └── results/                # raw logs, CSVs, plots
├── media/
│   └── demo_comparison.mp4    # side-by-side rollout clip
└── docs/
    └── writeup.md              # final report: method, results, failure analysis
```

---

## 5. Phased build plan

### Phase 0 — Environment setup
- Install LeRobot (`pip install lerobot` per official install guide), ManiSkill3, and confirm CUDA availability (`torch.cuda.is_available()`).
- Confirm `torch>=2.3.0` (SmolVLA requirement).
- **Exit criteria:** `lerobot-train --help` runs; ManiSkill3 example task renders.

### Phase 1 — Task and baseline
- Define a single, clearly-scoped pick-place task in ManiSkill3 (e.g., "pick up the cube and place it in the bin") with configurable object position/shape randomization.
- Build the scripted baseline policy (IK-based reach-grasp-place) for this task.
- **Exit criteria:** baseline policy completes the task at a known success rate over N trials at default (non-randomized) settings — this is your sanity-check number before anything ML-related happens.

### Phase 2 — Demonstration data collection
- Record ~50 demonstration episodes minimum, across 5+ distinct object position/pose variations (10 episodes each) — this ratio is what the SmolVLA authors recommend; fewer than ~25 episodes total is documented to underperform.
- Convert to LeRobot dataset format; validate every episode via replay (don't skip this — it's the most commonly reported failure point in community writeups).
- **Exit criteria:** dataset stats show balanced coverage across variations; replayed episodes visually match the recorded task.

### Phase 3 — Fine-tuning
- Run `lerobot-train` with `--policy.path=lerobot/smolvla_base`, your dataset, batch size tuned to available VRAM (start at 4–8), `--steps=20000` as a starting point, W&B logging on.
- Watch for the single most common failure mode: an **action-space mismatch** between how your sim reports robot state/actions and what SmolVLA expects — validate this before a long training run, not after.
- **Exit criteria:** training loss converges; checkpoint saved.

### Phase 4 — Evaluation suite
- In-distribution trials: same object positions/variations seen in training.
- Generalization trials: novel object positions (interpolated and extrapolated from training range), novel object shape/color, added distractor objects.
- Run both the fine-tuned VLA policy and the scripted baseline through identical trial sets.
- Log: success rate, grasp success rate (did it ever make contact/lift), time-to-completion, and a manual failure-mode tag (e.g., "missed grasp," "wrong object," "froze," "knocked object away").
- **Exit criteria:** a results table with both policies × both conditions, plus failure mode breakdown.

### Phase 5 — Writeup and demo
- `docs/writeup.md`: problem, method, results table, 2–3 sentence failure analysis per condition, what you'd try next (e.g., more demos, LoRA fine-tuning, larger base model).
- Record a short side-by-side clip of both policies on the same generalization trial.
- **Exit criteria:** README has the summary + gif at the top; nothing requires reading the writeup to get the headline result.

---

## 6. Known pitfalls (from community reports — check these before debugging blindly)

- **Action-space mismatch** is the most common silent failure: training converges fine, evaluation is garbage, because your sim's action/state representation doesn't match what the policy was trained to expect. Validate this explicitly in Phase 3 before a full training run.
- **Too few demonstrations, or not enough variation per episode set** — 25 episodes total has been reported as insufficient; go for 50+ with repeated coverage per variation, not 50 unrelated one-off demos.
- **Skipping replay validation** on collected data — always eyeball a replayed episode before training on it.
- Don't expect a first fine-tune to hit high success rates — even well-executed community attempts report contact/grasp rates around 40–90% depending on task difficulty, with full end-to-end success often lower. **A partial success rate with honest failure analysis is a legitimate, presentable result** — the comparison against baseline generalization is the actual point, not a leaderboard number.

---

## 7. CV bullet template (fill in your real numbers once done)

> **VLA Fine-Tuning for Robotic Manipulation (SmolVLA, LeRobot, ManiSkill3)**
> - Fine-tuned an open-source Vision-Language-Action model (SmolVLA, 450M params) on a custom pick-and-place demonstration dataset, achieving **[X]% task success rate** in-distribution
> - Designed a generalization evaluation suite (novel object positions/shapes, distractor objects) comparing the learned policy against a classical IK/motion-planning baseline, showing **[Y]% relative improvement in generalization / specific failure-mode tradeoffs**
> - Built end-to-end sim-to-policy pipeline: task environment, demonstration collection with replay validation, fine-tuning, and automated evaluation harness
