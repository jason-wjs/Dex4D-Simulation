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
- An intermittent `free(): invalid pointer` abort occurs in the native
  `gym.create_actor` call for the robot. One run passed scene creation and loaded
  the checkpoint; other attempts aborted before policy execution. The graphics
  fix does **not** establish that this allocation problem is fixed. Native
  debugging is ongoing; do not count these attempts as completed evaluations.

The actual diagnostic uses the original stage3 config/checkpoint, seed 0, four
training-list objects and the original two 400-step test rollouts. Four objects
are a setup fixture, not a benchmark. Preserve the unmodified original metrics.
