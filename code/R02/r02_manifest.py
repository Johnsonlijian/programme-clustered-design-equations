"""R02 manifest: hashes of every delivered artifact plus the environment actually used.

Run last: it records what shipped, not what was planned.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
INCLUDE = ("scripts/*.py", "tests/*.py", "outputs/*.csv", "outputs/*.csv.gz",
           "outputs/*.json", "outputs/*.txt", "figures/*.svg", "figures/*.pdf",
           "figures/preview/*.png", "note/*.tex", "note/*.pdf", "note/generated/*.tex")
SKIP = ("_scratch", "_patch", "_mintest")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def main():
    try:
        import numpy, pandas, scipy, sklearn, matplotlib  # noqa: F401
        pkgs = {"numpy": numpy.__version__, "pandas": pandas.__version__,
                "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                "matplotlib": matplotlib.__version__}
    except Exception:
        pkgs = {}

    files = []
    for pat in INCLUDE:
        for p in sorted(ROUND.glob(pat)):
            if p.is_file() and not any(s in p.name for s in SKIP):
                files.append(dict(path=str(p.relative_to(ROUND)).replace("\\", "/"),
                                  bytes=p.stat().st_size, sha256=sha256(p)))

    man = dict(
        round="R02_crossdomain_theory_2026-10-20",
        created_local=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        python=platform.python_version(),
        platform=platform.platform(),
        executable=sys.executable,
        packages=pkgs,
        note="R01 ran under a different interpreter (numpy 2.4.3 / pandas 3.0.3); R02 re-runs every arm "
             "inside one process, so all R02 comparisons are internally consistent.",
        files=files,
    )
    (ROUND / "RUN_MANIFEST.json").write_text(json.dumps(man, indent=1, ensure_ascii=False),
                                             encoding="utf-8")
    print(f"{len(files)} files hashed")


if __name__ == "__main__":
    main()
