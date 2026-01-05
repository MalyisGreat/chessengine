import argparse
import csv
import glob
import json
import os


def load_records(eval_dir: str):
    records = []
    for path in glob.glob(os.path.join(eval_dir, "**", "*.*"), recursive=True):
        if path.endswith((".jsonl", ".json")):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    rec["_source"] = path
                    records.append(rec)
        elif path.endswith(".csv"):
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["_source"] = path
                    records.append(row)
    return records


def main():
    parser = argparse.ArgumentParser(description="Aggregate eval JSON/CSV logs into a single JSONL file.")
    parser.add_argument("--eval-dir", required=True, help="Eval directory root")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    args = parser.parse_args()

    records = load_records(args.eval_dir)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {args.out} ({len(records)} rows)")


if __name__ == "__main__":
    main()
