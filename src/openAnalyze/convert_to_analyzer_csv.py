#!/usr/bin/env python3
"""
convert_to_analyzer_csv.py
──────────────────────────
Parse assorted detector dumps (TXT, CSV, TSV, ND-JSON, DeepMedia results.json)
and emit analyzer-ready CSVs, *inferring* fake/real labels from the input
filenames—unless the input itself is a 3-column CSV that already provides
true_label.

Label rule (unless overridden by a 3-col input):
──────────
•  If the raw input path (case-insensitive) contains “fake” → ground truth = FAKE (1)
•  If it contains “real” → ground truth = REAL (0)
•  Otherwise the file is skipped with a warning.

Outputs
───────
Default:  fake_data_<DET>.csv   and/or   real_data_<DET>.csv
          (columns: filename,score)

--combine: <DET>.csv   (columns: filename,score,true_label)

Usage example
─────────────
python3 convert_to_analyzer_csv.py \
        --input fake_dump.txt real_dump.txt new_three_col.csv \
        --detector genconvit \
        --out_dir csv_ready \
        [--combine]
"""
from __future__ import annotations
import argparse
import json
import csv
import sys
from pathlib import Path
from typing import Iterable, Tuple, List

# ────────────────────────────────────────────────────────────────────────────
#  Mini-parsers
# ────────────────────────────────────────────────────────────────────────────

def rows_from_semicolon_txt(path: Path) -> Iterable[Tuple[str, float]]:
    """
    Parse lines like:
      filename;score;[{"start":..., "end":..., "score":...}, …]
    We ignore the JSON‐“segments” and yield (filename, score).
    """
    with path.open(encoding="utf-8", errors="replace") as fh:
        header = fh.readline()  # usually "filename;score;segments..."
        for line in fh:
            parts = line.strip().split(";")
            if len(parts) >= 2:
                yield parts[0], float(parts[1])


def rows_from_two_col(path: Path, delim: str) -> Iterable[Tuple[str, float]]:
    """
    Parse lines like:
      filename⟷score     (two columns, delim is "," or "\t")
    """
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            parts = line.strip().split(delim)
            if len(parts) >= 2:
                yield parts[0], float(parts[1])


def rows_from_three_col(path: Path) -> Iterable[Tuple[str, float, int]]:
    """
    Parse lines like:
      filename,score,true_label  (three columns, comma-delimited).
    Yields (filename, float(score), int(true_label)).

    We expect exactly 3 comma-separated fields per line (no header).
    """
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            parts = raw.split(",", 2)
            if len(parts) != 3:
                # skip malformed lines
                continue
            fname = parts[0]
            try:
                score = float(parts[1])
                true_label = int(parts[2])
            except ValueError:
                # skip if score/label not parseable
                continue
            if true_label not in (0, 1):
                # only accept 0 or 1
                continue
            yield fname, score, true_label


def rows_from_json_lines(path: Path) -> Iterable[Tuple[str, float]]:
    """
    Parse ND-JSON lines where each line is a JSON object containing:
      { "filename": ..., "score": ... }
    """
    with path.open(encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            obj = json.loads(raw)
            fname = obj.get("filename") or obj.get("path") or obj.get("file")
            if fname is None or "score" not in obj:
                continue
            yield fname, float(obj["score"])


def rows_from_deepmedia_results(path: Path) -> Iterable[Tuple[str, float]]:
    """
    Parse a DeepMedia results.json (could be either a single object or a list).
    We look for fields "pred" inside each trajectory's "results" sub-dictionary,
    average them, and yield (video_id, mean_score).
    """
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(data, list):
        data = [data]

    scores: List[float] = []
    for traj in data:
        for res in traj.get("results", {}).values():
            if isinstance(res.get("pred"), (int, float)):
                scores.append(float(res["pred"]))

    vid_id = path.parent.stem
    mean_score = (sum(scores) / len(scores)) if scores else 0.0
    yield vid_id, mean_score


def sniff_rows(path: Path) -> Iterable[Tuple[str, float]]:
    """
    Auto-detect format and yield (filename, score). Raises ValueError if unknown.
    NOTE: This function never yields a true_label; it’s only meant for
          2-col / semicolon‐style / JSON‐ND / DeepMedia results.json.
    If you need to handle a 3-col CSV (filename,score,true_label), handle that
    before calling sniff_rows().
    """
    name = path.name.lower()

    # DeepMedia “results.json” (list or single JSON object)
    if name == "results.json":
        return rows_from_deepmedia_results(path)

    # Any other *.json → ND-JSON lines
    if path.suffix.lower() == ".json":
        return rows_from_json_lines(path)

    # Read first line to sniff delimiters
    try:
        head = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    except Exception as e:
        raise ValueError(f"Cannot read {path}: {e}") from e

    # semicolon-CSV, header starts with "filename;score"
    if head.lower().startswith("filename;score"):
        return rows_from_semicolon_txt(path)

    # exactly one comma → two-column CSV
    if head.count(",") == 1:
        return rows_from_two_col(path, ",")

    # exactly one tab → two-column TSV
    if head.count("\t") == 1:
        return rows_from_two_col(path, "\t")

    raise ValueError(f"Unrecognized format for {path}")


# ────────────────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument(
        "--input", required=True, nargs="+",
        help="Raw file(s) or directory(ies) to convert")
    ap.add_argument(
        "--detector", required=True,
        help="Detector token to embed in output filenames")
    ap.add_argument(
        "--out_dir", default="csv_out",
        help="Where to write the CSV(s)")
    ap.add_argument(
        "--combine", action="store_true",
        help="Emit ONE combined CSV with true_label column")
    args = ap.parse_args()

    # 1) Gather all file paths (expand directories recursively) ---------------
    in_paths: List[Path] = []
    for p in args.input:
        ppath = Path(p).expanduser()
        if ppath.is_dir():
            in_paths.extend([q for q in ppath.rglob("*") if q.is_file()])
        elif ppath.is_file():
            in_paths.append(ppath)
        else:
            print(f"⚠️  Skipping non-existent path: {ppath}", file=sys.stderr)

    if not in_paths:
        sys.exit("No input files found.")

    # 2) Accumulate rows, bucketed by label ----------------------------
    rows_fake: List[Tuple[str, float]] = []
    rows_real: List[Tuple[str, float]] = []
    rows_combined: List[Tuple[str, float, int]] = []  # only used if --combine

    for path in in_paths:
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if not text:
            print(f"⚠️  Empty file, skipping: {path}", file=sys.stderr)
            continue

        head = text[0].strip()

        # 2a) If it looks like a 3‐column CSV ("filename,score,true_label"), handle it directly:
        #     - header might be absent; we just look for exactly two commas on each line
        if head.count(",") == 2 and all(line.count(",") == 2 for line in text if line.strip()):
            # parse every non-empty line as (fname, score, true_label)
            try:
                for fname, score, true_label in rows_from_three_col(path):
                    if true_label == 1:
                        rows_fake.append((fname, score))
                    else:
                        rows_real.append((fname, score))
                    rows_combined.append((fname, score, true_label))
            except Exception as e:
                print(f"⚠️  Error parsing 3‐col CSV {path}: {e}", file=sys.stderr)
            continue

        # 2b) Otherwise, infer label from the path ("fake" or "real").
        low = str(path).lower()
        if "fake" in low and "real" not in low:
            inferred_label = 1
        elif "real" in low and "fake" not in low:
            inferred_label = 0
        else:
            print(f"⚠️  Could not infer label for {path} (needs 'fake' or 'real' in path)", file=sys.stderr)
            continue

        # Now use sniff_rows() to get (filename, score), then append to appropriate bucket.
        try:
            for fname, score in sniff_rows(path):
                if inferred_label == 1:
                    rows_fake.append((fname, score))
                    rows_combined.append((fname, score, 1))
                else:
                    rows_real.append((fname, score))
                    rows_combined.append((fname, score, 0))
        except ValueError as e:
            print(f"⚠️  {e}", file=sys.stderr)

    if not (rows_fake or rows_real):
        sys.exit("No rows parsed — check formats / filenames.")

    # 3) Write outputs -----------------------------------------------------------
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.combine:
        # Single CSV with columns: filename, score, true_label
        out_csv = out_dir / f"{args.detector}.csv"
        with out_csv.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["filename", "score", "true_label"])
            for fname, score, label in rows_combined:
                writer.writerow([fname, score, label])
        total = len(rows_combined)
        print(f"✓ Wrote combined CSV → {out_csv}  ({total} rows)")

    else:
        # Split into fake_data_<DET>.csv and real_data_<DET>.csv
        if rows_fake:
            fake_csv = out_dir / f"fake_data_{args.detector}.csv"
            with fake_csv.open("w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["filename", "score"])
                writer.writerows(rows_fake)
            print(f"✓ Wrote {fake_csv}  ({len(rows_fake)} rows)")

        if rows_real:
            real_csv = out_dir / f"real_data_{args.detector}.csv"
            with real_csv.open("w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["filename", "score"])
                writer.writerows(rows_real)
            print(f"✓ Wrote {real_csv}  ({len(rows_real)} rows)")


if __name__ == "__main__":
    main()
