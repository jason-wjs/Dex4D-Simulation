"""Fetch the official UniDexGrasp sample, not the full evaluation dataset.

Run with Python >=3.8 from this checkout. No simulator or GPU required.
The car appears in Dex4D's training split; this only enables runtime validation.
"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

SOURCE = "PKU-EPIC/UniDexGrasp"
COMMIT = "36c9bfcf7cedc987dd15e2384c548e51d5aadc21"
PREFIX = "dexgrasp_policy/assets/"
ROOT = Path(__file__).resolve().parents[1]


def main():
    with urlopen("https://api.github.com/repos/" + SOURCE + "/git/trees/" + COMMIT + "?recursive=1", timeout=30) as response:
        tree = json.load(response)
    entries = [e for e in tree["tree"] if e["type"] == "blob" and
               e["path"].startswith((PREFIX + "meshdatav3_scaled/", PREFIX + "meshdatav3_pc_feat/"))]

    def download(entry):
        url = "https://raw.githubusercontent.com/" + SOURCE + "/" + COMMIT + "/" + entry["path"]
        with urlopen(url, timeout=60) as response:
            data = response.read()
        blob = ("blob %d\0" % len(data)).encode() + data
        if hashlib.sha1(blob).hexdigest() != entry["sha"]:
            raise ValueError("Git blob mismatch: " + entry["path"])
        path = ROOT / "dex4d_policy/assets" / entry["path"][len(PREFIX):]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(data).hexdigest()}

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(download, entries))
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    (out / "smoke-assets.json").write_text(json.dumps({"source": SOURCE, "commit": COMMIT, "files": records}, indent=2) + "\n")
    print("Verified and installed %d sample files; full dataset still required." % len(records))


if __name__ == "__main__":
    main()
