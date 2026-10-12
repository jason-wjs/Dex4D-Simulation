# Reproduction branch

Upstream reference: `Dex4D/Dex4D-Simulation` at
`0dfccd82797e929cda6ab665003bc27a6b8cd31a`.

Preserve xArm6 + LEAP, AP2AP teacher PPO/PPE, the three curriculum stages and
the subsequent DAgger student. Start with official Isaac Gym pretrained playback
and finite evaluation. Isaac Lab and mjlab implementations will be explicit ports.
The sibling `jason-wjs/dex-grasp-study` repository owns experiment protocols,
source/checkpoint hashes and comparative reports.

The tracked upstream checkpoints are available; the external UniDexGrasp object
dataset and compiled simulator/point-cloud extensions are separate prerequisites.
Do not replace missing assets with unrelated geometry and call it a reproduction.
Keep downloads, environments, new checkpoints and experiment outputs untracked.

## Dataset acquisition and checks

The official mirror's directory page currently renders the mirror homepage, but
its files are accessible directly at:

- `https://mirrors.pku.edu.cn/dl-release/UniDexGrasp_CVPR2023/dexgrasp_policy/assets/meshdatav3_scaled.tar.xz`
- `https://mirrors.pku.edu.cn/dl-release/UniDexGrasp_CVPR2023/dexgrasp_policy/assets/meshdatav3_pc_feat.zip`

Downloaded on 2026-10-12; SHA256 respectively:

```text
18786d34c14472a20ec962d67bf938377a8ab7b035840868d6cf9d206d13b9af
c4d7e6b55f956faa8538e731bf30e3fc178e10b04365d02c9c8794099ab7f416
```

Extract both under `dex4d_policy/assets`, then run:

```bash
python reproduction/audit_dataset.py --output outputs/full-dataset-coverage.json
```

The training split contains 3,200 object-scale entries and 2,145 unique meshes.
Do not collapse repeated YAML keys: upstream `MergeLoader` merges scale lists.
The two test lists are mapping fragments, not standalone task config files.
The audit handles these exact source formats and exits nonzero on missing assets.

`download_smoke_assets.py` fetches the original one-object UniDexGrasp example
with Git-blob verification. It is useful for setup only, not full evaluation.

## Observed runtime issues

The official Python 3.8.20 standalone runtime lacks `xxlimited`, a CPython example
extension that both task modules imported but never used. This branch removes
only those imports; task logic and network behavior are unchanged. Full native
playback is still required to validate the remaining imports and simulator.

## Validated native runtime on Tele

Python 3.8.20, torch 2.0.1+cu118, torchvision 0.15.2+cu118,
torchaudio 2.0.2+cu118 and NumPy 1.23.5 are installed in `.venv-gym`.
Official Isaac Gym Preview 4 lives at `../runtime/isaacgym`.
Install the author packages without changing those core pins:

```bash
uv --no-config pip install --python .venv-gym/bin/python \
  -e dex4d_policy -e ../runtime/isaacgym/python \
  'torch==2.0.1+cu118' 'numpy==1.23.5' ninja \
  --index https://pypi.tuna.tsinghua.edu.cn/simple \
  --default-index https://pypi.org/simple
export PATH="$PWD/.venv-gym/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/.python/cpython-3.8.20-linux-x86_64-gnu/lib"
export CUDA_VISIBLE_DEVICES=0 MAX_JOBS=2 OMP_NUM_THREADS=4
cd dex4d_policy/dex4d
../../.venv-gym/bin/python train.py --help
```

Check GPU availability before choosing `CUDA_VISIBLE_DEVICES`. The library path
above is for our standalone interpreter; use your interpreter's own library
directory elsewhere. `tb-nightly` requires canonical PyPI because it is absent
from the mirror. Native `gymapi` import, gymtorch compilation and the teacher CLI
passed. Teacher imports do not require the README's optional PointNet2/PyTorch3D
builds; student paths are not yet validated.

The first headless teacher attempts revealed two separate failures:

- With graphics disabled, native test code still creates a camera and reshapes
  an empty image. This branch skips PPO recording and teacher table-texture
  operations when the existing `graphics_device_id` is negative. It does not
  change action, reward, observation, reset or success calculations.
- An intermittent `free(): invalid pointer` abort was observed in the native
  `gym.create_actor` call for the robot. One run passed scene creation and loaded
  the checkpoint; other attempts aborted before policy execution. The graphics
  fix does **not** establish that this allocation problem is fixed. Native
  debugging subsequently located the aggregate allocation mismatch documented
  below. Those failed attempts do not count as completed evaluations.

The actual diagnostic uses the original stage3 config/checkpoint, seed 0, four
training-list objects and the original two 400-step test rollouts. Four objects
are a setup fixture, not a benchmark. Preserve the unmodified original metrics.

## Preview 4 GPU aggregate allocation repair

A minimal scene with the original robot reproduces the native abort without
PyTorch, PPO, object meshes or checkpoints. Removing force sensors did not fix it.
Valgrind located a host heap write past a 24-byte allocation in
`physx::PxgAABBManager::createAggregate`, called from `gym.begin_aggregate`.
The later `create_actor`/`create_env` aborts were detection sites.

Inspection of the pinned Preview 4 `libPhysXGpu_64.so` routine at `0x6f2da–0x6f356`
shows bitmap allocation `4 * ((n + 31) >> 5)` bytes, but initialization
`(n + 31) >> 3` bytes. For shape capacity 179 this allocates 24 and clears 26 bytes.
Rounding to a multiple of 32 is **not** a fix: 256 allocates 32 and clears 35,
even if allocator padding happens to hide the crash. A capacity congruent to
1 modulo 32 makes both lengths agree. We increase the original shape capacity to
`((n + 30) // 32) * 32 + 1` (179 becomes 193: 28 bytes for both operations).

This alters reserved capacity only. It keeps the original aggregate grouping,
self-collision flag, robot, all shapes, force sensors, solver and task semantics.
The vision task inherits this scene creation. No student validation is implied.
The pinned simulator binary is not modified.

Native regression fixture (outside training; choose an idle GPU first):

```bash
CUDA_VISIBLE_DEVICES=0 timeout 90 .venv-gym/bin/python \
  reproduction/check_aggregate_capacity.py --max-shapes 179
CUDA_VISIBLE_DEVICES=0 timeout 90 .venv-gym/bin/python \
  reproduction/check_aggregate_capacity.py --max-shapes 193
```

The first command deliberately exercises the original native failure. Inspect
memory-checker output, not merely exit status: the original scene can reach exit
0 under Valgrind while still reporting the concrete invalid write. Detailed
runtime validation and raw-log locations are maintained in the study repository.

The repaired teacher completed the original two 400-step test rollouts on H100,
GPU PhysX and GPU tensors, four objects and seed 0: successes 1.0 / 1.0,
current_successes 1.0 / 1.0, consecutive_successes 3.25 / 3.5 (exit 0).
These four training-list objects are not an unseen-object benchmark. Replay with:

```bash
# At repo root, after setting the Gym runtime exports above:
mkdir -p outputs/teacher-stage3
cp dex4d_policy/dex4d/example_models/teacher_policy_stage_3/model_34000_best.pt outputs/teacher-stage3/
model="$PWD/outputs/teacher-stage3/model_34000_best.pt"
cd dex4d_policy/dex4d
timeout --kill-after=15s 300 ../../.venv-gym/bin/python train.py \
  --task XArm6LeapHandAP2AP --algo ppo --seed 0 \
  --rl_device cuda:0 --sim_device cuda:0 --test --num_envs 4 --headless \
  --model_dir "$model" --vis_env_id 0 \
  --cfg_env cfg/xarm6_leap_hand_ap2ap_stage_3.yaml \
  --cfg_train cfg/ppo/config_stage_3.yaml
```

Targeted memory regression passed: original capacity 179 reports one aggregate
invalid-write context; repaired capacity 193 reports none and exits normally.
Other CPython/CUDA Valgrind reports remain, so this is not a globally clean
memory-check claim. The rounding leaves already-conforming capacities unchanged.

## Actual evaluation coverage

Test invocations now save `evaluation_cases.json` beside the supplied checkpoint,
in environment order, directly from the instantiated task's object/scale list.
Use a distinct checkpoint output directory per evaluation so the manifest and
the author's append-only metrics remain attributable to that run.

The original test filter retains only the first scale for each selected mesh.
Additionally, its scene assignment uses `int(i / num_envs * num_objects)`;
with 50 environments and 50 selected meshes, floating-point rounding maps
environment 29 to object 28. The observed manifest therefore has 50 entries but
49 unique pairs, including `core/mug-962883677a586bd84a60c1a189046dd1` / `010`
twice (environments 28 and 29). This branch records that original behavior;
it does not silently change the mapping or claim complete split coverage.

## Complete held-out split evaluation

The optional test-only `env.evaluation_object_pairs` assigns one ordered
`[mesh_code, scale]` entry to each environment. It bypasses the original first
scale filter and floating-point index mapping; omitting it preserves the original
training and playback path. Every requested scale is loaded, while repeated
identical entries remain separate environments. `train.py` verifies the actual
case manifest before loading the policy. The task's reset, goal sampling, reward,
success conditions, physics and network are unchanged.

The official seen-category list contains 141 entries / 141 unique pairs / 137
meshes. The unseen-category list contains 100 entries / 78 unique pairs / 43
meshes. Preserve duplicate YAML keys and repeated entries; a normal YAML mapping
loader can silently remove cases. This is an explicitly complete-split protocol,
distinct from the author's default first-50-training-mesh playback.

From the fork root, on an idle GPU with the isolated environment installed:

```bash
.venv-gym/bin/python reproduction/prepare_split_eval.py \
  --split test_set_unseen_cat --output outputs/unseen-eval
.venv-gym/bin/python reproduction/run_split_eval.py \
  --output outputs/unseen-eval --gpu 0 --max-seconds 1200
```

Use `test_set_seen_cat` and a separate output directory for the other split.
The runner uses the original stage3 checkpoint and seed 0, records source/config
hashes and argv, checks the GPU every ten seconds, and stops only its process
group on a time limit or foreign GPU process. It verifies exact case order,
two original metric snapshots and agreement between native means and per-env
values. An incomplete process is not accepted as a completed evaluation.

Teacher test playback additionally writes `evaluation_rollouts.json` beside the
checkpoint, retaining the author's three metric names and reporting steps.
Per-environment arrays correspond to `evaluation_cases.json` order. Read-only
recording does not change metric computation. The study repository's
`experiments/dex4d/summarize_split_eval.py` independently checks the original
split file and reports entry-weighted and unique-pair-weighted results; the latter
first averages repeated entries of a pair. Neither statistic certifies contact
quality or replaces the native metric with a grasp-success definition.
