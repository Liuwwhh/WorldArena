# WorldArena Challenge — Track 2 Submission Template

> **CVPR 2026 WorldArena Challenge** — [Challenge Page](http://cvpr2026challenge.world-arena.ai/)

Track 2 evaluates **world models** in closed-loop with a fixed policy. You provide the world model; we provide the policy, dataset, and evaluation pipeline.

📖 For detailed technical documentation (dataset format, action space, bridge, rollout internals), see **[docs/DETAILS.md](docs/DETAILS.md)**.

---

## Step 1 · Environment Setup

### 1a. Policy environment ([openpi](https://github.com/Physical-Intelligence/openpi))

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### 1b. Download policy checkpoints (5 variants)

```bash
huggingface-cli download WorldArena/WorldArena \
  --repo-type model --local-dir ./policy_ckpt
```

This downloads `10data/`, `20data/`, `30data/`, `50data/`, `fulldata/` — each containing `model.safetensors`, `metadata.pt`, and norm stats.

### 1c. Download dataset (500 episodes, ≈21 MB)

```bash
bash scripts/download_dataset.sh
# or manually:
wget https://huggingface.co/datasets/WorldArena/WorldArena_Robotwin2.0/resolve/main/dataset.tar.gz
tar -xzf dataset.tar.gz
```

---

## Step 2 · Write Your Adapter

Create a new adapter in `src/worldarena_track2_template/adapters/`. See the two examples:

- `example_joint14.py` — for WMs trained on **joint angles** (no bridge needed)
- `example_endpose14.py` — for WMs trained on **end-effector poses** (needs kNN bridge)

Your adapter just needs to implement `build_command()` — it returns the shell command to run your WM rollout script.

> **Key decision:** Does your WM use `joint14` or `endpose14` action space?
> - **joint14** → set `bridge_mode = "passthrough"`
> - **endpose14** → set `bridge_mode = "task_knn"` (see [docs/DETAILS.md](docs/DETAILS.md#world-model-action-space--bridge) for why)

---

## Step 3 · Run Generation (5 policies × 500 episodes)

```bash
for variant in 10data 20data 30data 50data fulldata; do
  python scripts/run_generation.py \
    --wm <your_wm> \
    --dataset-root ./dataset \
    --output-dir ./output/${variant} \
    --policy-variant ${variant} \
    --max-episode-index 500
done
```

This produces **2500 videos** (5 folders × 500 episodes each).

---

## Step 4 · Package & Submit

### 4a. Fill in `model_README.md`

Copy and edit `examples/model_README.template.md`:

| Field | Example |
|-------|---------|
| Model Name | `my_world_model` |
| Action Space | `joint14` or `endpose14` |
| Open Source | `Yes` / `No` |

### 4b. Package

```bash
python scripts/package_submission.py \
  --model-name my_model \
  --model-readme my_model_README.md \
  --video-dirs ./output/10data ./output/20data ./output/30data ./output/50data ./output/fulldata \
  --output ./my_model_eval.zip
```

### 4c. Submit

Email `my_model_eval.zip` to **WorldArena1@outlook.com**.

The archive structure will be:

```text
my_model_eval/
├── my_model_10data/    (500 videos)
├── my_model_20data/    (500 videos)
├── my_model_30data/    (500 videos)
├── my_model_50data/    (500 videos)
├── my_model_fulldata/  (500 videos)
└── model_README.md
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
