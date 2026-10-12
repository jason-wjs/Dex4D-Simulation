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
