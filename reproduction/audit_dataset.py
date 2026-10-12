"""Verify mesh/scale coverage without collapsing duplicate YAML object keys."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def split_pairs(root, split):
    path = root / "dex4d_policy/dex4d/cfg" / (split + ".yaml")
    text = path.read_text()
    if split == "train_set":
        document = yaml.compose(text)
        mapping = next(value for key, value in document.value if key.value == "object_code_dict")
    else:
        # The shipped test lists are mapping fragments, not task config files.
        mapping = yaml.compose("{" + text + "}")
    # Preserve repeated mesh keys, as upstream MergeLoader does.
    pairs = [(key.value, float(scale.value)) for key, values in mapping.value for scale in values.value]
    return path, pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    assets = root / "dex4d_policy/assets"
    reports = {}
    for split in ("train_set", "test_set_seen_cat", "test_set_unseen_cat"):
        path, pairs = split_pairs(root, split)
        missing = []
        for code, scale in pairs:
            tag = f"{round(scale * 100):03d}"
            paths = [assets / "meshdatav3_pc_feat" / code / f"pc_feat_{tag}.npy",
                     assets / "meshdatav3_scaled" / code / "coacd" / f"coacd_{tag}.urdf",
                     assets / "meshdatav3_scaled" / code / "coacd" / f"decomposed_{tag}.obj"]
            missing.extend(str(p.relative_to(root)) for p in paths if not p.is_file())
        reports[split] = {"config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "entries": len(pairs), "unique_meshes": len({code for code, _ in pairs}),
                          "missing_paths": missing}
        print(split, "entries=", len(pairs), "unique_meshes=", reports[split]["unique_meshes"],
              "missing_files=", len(missing))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2) + "\n")
    if any(report["missing_paths"] for report in reports.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
