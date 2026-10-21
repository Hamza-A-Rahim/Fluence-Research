"""Fill in the exact model version in existing embedding manifests.

Embeddings extracted before this fix recorded "revision": null. The exact
checkpoint commit is still in the local Hugging Face cache; this copies it in.

    python scripts/record_model_revisions.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from extract_embeddings import model_revision  # noqa: E402

for manifest in sorted((ROOT / "data/processed/embeddings").glob("*/manifest.json")):
    m = json.loads(manifest.read_text(encoding="utf-8"))
    rev = model_revision(m["model"])
    if rev and not m.get("revision"):
        m["revision"] = rev
        manifest.write_text(json.dumps(m, indent=2), encoding="utf-8")
    print(f"{m['model']}: {m.get('revision') or 'NOT FOUND in cache'}")
