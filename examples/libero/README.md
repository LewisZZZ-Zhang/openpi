# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

## With Docker (recommended)

```bash
# Grant access to the X11 server:
sudo xhost +local:docker

# To run with the default checkpoint and task suite:
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

You can customize the loaded checkpoint by providing additional `SERVER_ARGS` (see `scripts/serve_policy.py`), and the LIBERO task suite by providing additional `CLIENT_ARGS` (see `examples/libero/main.py`).
For example:

```bash
# To load a custom checkpoint (located in the top-level openpi/ directory):
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"

# To run the libero_10 task suite:
export CLIENT_ARGS="--args.task-suite-name libero_10"
```

## Without Docker (not recommended)

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the simulation
python examples/libero/main.py

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx python examples/libero/main.py
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env LIBERO
```

## Seed-123 LIBERO-100 native pi0-FAST baseline

The `pi0_fast_libero100_seed123` config trains on the same 70 tasks and 40
query demonstrations per task as the RICL experiment. It reads the original
LIBERO HDF5 files through the shared corpus metadata, but does not load
retrieval embeddings, neighbours, or context demonstrations.

From the VFE repository root, use the unified Slurm launcher:

```bash
# One H200 allocation; train first, then evaluate 30 unseen tasks x 50 rollouts.
MODE=train_eval EXP_NAME=libero100_pi0_fast_seed123 OVERWRITE=1 \
  bash shells/run_pi0_fast_libero100_h200.sh

# Training or evaluation only.
MODE=train EXP_NAME=libero100_pi0_fast_seed123 RESUME=1 \
  bash shells/run_pi0_fast_libero100_h200.sh
MODE=eval EXP_NAME=libero100_pi0_fast_seed123 EVAL_CHECKPOINT_STEP=10000 \
  bash shells/run_pi0_fast_libero100_h200.sh
```

The launcher creates normalization statistics when they are missing, verifies
them against the RICL statistics, and evaluates through one policy server plus
four task-sharded LIBERO clients. The merged result is written to
`outputs/pi0_fast_libero100/<experiment>/<step>/unseen50/metrics.json`.

## Results

If you want to reproduce the following numbers, you can evaluate the checkpoint at `gs://openpi-assets/checkpoints/pi05_libero/`. This
checkpoint was trained in openpi with the `pi05_libero` config.

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85
