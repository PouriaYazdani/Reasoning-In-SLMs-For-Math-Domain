# one_time_collect_families.py
# One-time run, no CLI args.
# Collects ALL rows whose family_id is in the hardcoded list from multiple out_* folders,
# and writes them to a single JSONL.

from __future__ import annotations

import json
from pathlib import Path


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    # ===== HARD CODE HERE =====
    ROOT = Path(".").resolve()
    OUT_FILE = ROOT / "api_call" / "40_train.jsonl"

    OUT_DIRS = [
        "out_stat",
        "out_algebra",
        "out_geometry",
        "out_nt_fin",
        "out_calc",
        "out_dmopt",
    ]

    # Collect ALL instances for these families (across all scanned folders)
    FAMILIES = {
        "f01","f02","f03","f04","f05","f06","f07","f08","f09","f10",#"f11","f12",
        "f13","f14","f15","f16","f17","f18","f19","f20",#"f21","f22",
        "f23","f24","f25","f26","f27","f28",#"f29","f30",
        "f31","f32","f33","f34",#"f35",
        "f36","f37","f38","f39",#"f40",
        "f41","f42","f43","f44","f45","f46","f47","f48",#"f49","f50",
    }
    # =========================

    out_records: list[dict] = []

    for out_name in OUT_DIRS:
        od = ROOT / out_name
        probs = od / "problems.jsonl"
        if not probs.exists():
            print(out_name)
            continue

        for row in iter_jsonl(probs):
            fid = row.get("family_id")
            if fid in FAMILIES:
                out_records.append(
                    {
                        "folder": od.name,
                        "family_id": fid,
                        "text": row.get("text", ""),
                        # keep identifiers if present in your format
                        "problem_id": row.get("problem_id", ""),
                    }
                )

    if not out_records:
        raise RuntimeError("No matching rows found for the hardcoded families.")

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out_records)} rows to: {OUT_FILE}")

if __name__ == "__main__":
    main()