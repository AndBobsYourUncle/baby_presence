#!/usr/bin/env python3
"""Bulk-label frames by time range.

Reads a CSV of (start, end, label) ranges and copies/links frames from
FRAMES_DIR into OUT_DIR/occupied/ and OUT_DIR/empty/ based on the
timestamp encoded in each filename.

Frame filenames are produced by main.py and look like:
    YYYYMMDD-HHMMSS_<yolo_guess>_c<conf>.jpg

The YOLO guess portion is ignored — this script labels purely by time.

CSV format (header required). Timestamps accept either ISO format
(2026-04-21T08:00:00) or the filename format (20260421-080000):
    start,end,label
    20260421-080000,20260421-101500,occupied
    20260421-103000,20260421-120000,empty

Frames outside any range are skipped. A buffer (default 120s) is trimmed
from each side of every range so transition moments (lifting him in/out,
leaning over the crib) don't pollute training data.
"""
import argparse
import csv
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path

FRAME_RE = re.compile(r"(\d{8}-\d{6})_.+\.jpg$")
VALID_LABELS = {"occupied", "empty"}


def parse_timestamp(s: str) -> datetime:
    s = s.strip()
    try:
        return datetime.strptime(s, "%Y%m%d-%H%M%S")
    except ValueError:
        return datetime.fromisoformat(s)


def parse_ranges(csv_path: Path, buffer_seconds: int):
    ranges = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            start = parse_timestamp(row["start"])
            end = parse_timestamp(row["end"])
            label = row["label"].strip().lower()
            if label not in VALID_LABELS:
                raise ValueError(
                    f"invalid label {label!r} (expected one of {VALID_LABELS})"
                )
            start += timedelta(seconds=buffer_seconds)
            end -= timedelta(seconds=buffer_seconds)
            if start >= end:
                print(f"  skipping range — buffer ate it: {row}")
                continue
            ranges.append((start, end, label))
    return ranges


def frame_timestamp(path: Path) -> datetime | None:
    m = FRAME_RE.match(path.name)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d-%H%M%S")


def label_for(ts: datetime, ranges) -> str | None:
    for start, end, label in ranges:
        if start <= ts <= end:
            return label
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", required=True, type=Path, help="directory of .jpg frames")
    p.add_argument("--ranges", required=True, type=Path, help="CSV with start,end,label")
    p.add_argument("--out", required=True, type=Path, help="output dataset directory")
    p.add_argument(
        "--buffer", type=int, default=120,
        help="seconds to trim from each side of every range (default: 120)",
    )
    p.add_argument(
        "--link", action="store_true",
        help="symlink instead of copying (faster, no extra disk)",
    )
    args = p.parse_args()

    ranges = parse_ranges(args.ranges, args.buffer)
    if not ranges:
        print("no usable ranges after applying buffer")
        return 1

    print(f"loaded {len(ranges)} range(s):")
    for s, e, l in ranges:
        print(f"  {s} .. {e}  -> {l}")

    for sub in ("occupied", "empty"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    counts = {"occupied": 0, "empty": 0, "skipped": 0, "unparsable": 0}
    for frame in sorted(args.frames.glob("*.jpg")):
        ts = frame_timestamp(frame)
        if ts is None:
            counts["unparsable"] += 1
            continue
        label = label_for(ts, ranges)
        if label is None:
            counts["skipped"] += 1
            continue

        dest = args.out / label / frame.name
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        if args.link:
            dest.symlink_to(frame.resolve())
        else:
            shutil.copy2(frame, dest)
        counts[label] += 1

    print()
    print(f"occupied:   {counts['occupied']}")
    print(f"empty:      {counts['empty']}")
    print(f"skipped:    {counts['skipped']} (outside any range)")
    if counts["unparsable"]:
        print(f"unparsable: {counts['unparsable']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
