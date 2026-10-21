"""Check which SEP-28k / FluencyBank episode audio URLs still respond.

This makes HEAD requests only — it downloads no audio. Podcast hosting
changes over time, so some of the 385 SEP-28k episode links may have died
since 2021. Knowing how many survive is the first fact we need before
deciding whether a reproduction on SEP-28k is feasible.

Usage:
    python scripts/check_sep28k_availability.py data/metadata/SEP-28k_episodes.csv \
        --out data/metadata/sep28k_url_status.csv

Get the episodes CSV (annotations/metadata, CC BY-NC 4.0) from
https://github.com/apple-aiml-research/ml-stuttering-events-dataset
The CSV has no header; columns are: show, episode label, audio URL,
show tag, episode index.
"""

from __future__ import annotations

import argparse
import csv
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

UA = "fluence-research availability check (HEAD only; non-commercial research)"


def head(url: str, timeout: float = 15.0) -> tuple[str, str]:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return str(r.status), r.headers.get("Content-Length", "")
    except urllib.error.HTTPError as e:
        return str(e.code), ""
    except Exception as e:  # DNS failure, timeout, TLS error ...
        return type(e).__name__, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes_csv", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--delay", type=float, default=0.5, help="seconds between requests (be polite)")
    args = ap.parse_args()

    rows = list(csv.reader(args.episodes_csv.open(encoding="utf-8")))
    results, tally = [], Counter()
    for i, row in enumerate(rows, 1):
        show, episode, url, tag, idx = [c.strip() for c in row[:5]]
        status, size = head(url)
        ok = status.startswith("2") or status.startswith("3")
        tally[(tag, ok)] += 1
        results.append({"show_tag": tag, "ep_index": idx, "url": url,
                        "status": status, "content_length": size, "available": ok})
        print(f"[{i}/{len(rows)}] {tag} {idx}: {status}")
        time.sleep(args.delay)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)

    print("\nAvailable / total per show:")
    for tag in sorted({t for t, _ in tally}):
        ok, bad = tally[(tag, True)], tally[(tag, False)]
        print(f"  {tag:<14} {ok:>4} / {ok + bad}")
    total_ok = sum(v for (t, ok), v in tally.items() if ok)
    print(f"  {'TOTAL':<14} {total_ok:>4} / {len(rows)}")


if __name__ == "__main__":
    main()
