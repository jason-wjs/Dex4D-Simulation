"""Extract the original train/test files used by AP2AP from verified archives.

Avoid hundreds of thousands of unused files on shared storage. Full archives
remain in outputs/downloads. Run audit_dataset.py after this command.
"""

import hashlib
import tarfile
import zipfile
from pathlib import Path

from audit_dataset import split_pairs

ROOT = Path(__file__).resolve().parents[1]


def main():
    meshes, features = set(), set()
    for split in ("train_set", "test_set_seen_cat", "test_set_unseen_cat"):
        _, pairs = split_pairs(ROOT, split)
        for code, scale in pairs:
            tag = f"{round(scale * 100):03d}"
            meshes.update([f"meshdatav3_scaled/{code}/coacd/coacd_{tag}.urdf",
                           f"meshdatav3_scaled/{code}/coacd/decomposed_{tag}.obj"])
            features.add(f"meshdatav3_pc_feat/{code}/pc_feat_{tag}.npy")
    downloads = ROOT / "outputs/downloads"
    assets = ROOT / "dex4d_policy/assets"
    expected = {
        "meshdatav3_scaled.tar.xz": "18786d34c14472a20ec962d67bf938377a8ab7b035840868d6cf9d206d13b9af",
        "meshdatav3_pc_feat.zip": "c4d7e6b55f956faa8538e731bf30e3fc178e10b04365d02c9c8794099ab7f416",
    }
    for name, digest in expected.items():
        h = hashlib.sha256()
        with (downloads / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() != digest:
            raise ValueError("Archive SHA256 mismatch: " + name)
    with tarfile.open(downloads / "meshdatav3_scaled.tar.xz", "r|xz") as archive:
        for member in archive:
            if member.name in meshes:
                path = assets / member.name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.extractfile(member).read())
                meshes.remove(member.name)
    if meshes:
        raise ValueError("Missing geometry archive members: " + repr(sorted(meshes)))
    with zipfile.ZipFile(downloads / "meshdatav3_pc_feat.zip") as archive:
        for name in sorted(features):
            path = assets / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(name))
    print("Extracted original train/test geometry and features; run coverage audit next.")


if __name__ == "__main__":
    main()
