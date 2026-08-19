#!/usr/bin/env python3
"""Aggregate sanity-check JSON/CSV files over repeated random seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def identity(row):
    return tuple((key, str(row.get(key, ""))) for key in (
        "condition", "method", "threshold", "threshold_value", "donor_dataset"
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-json", nargs="+", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.seed_json) != len(args.seeds):
        parser.error("--seed-json and --seeds must have the same length")
    payloads = [json.loads(path.read_text()) for path in args.seed_json]
    indexed = [{identity(row): row for row in payload["results"]} for payload in payloads]
    keys = set(indexed[0])
    if any(set(index) != keys for index in indexed[1:]):
        raise ValueError("Seed sanity results have inconsistent conditions")

    records = []
    for key in sorted(keys):
        rows = [index[key] for index in indexed]
        record = {
            name: value for name, value in rows[0].items()
            if not isinstance(value, (int, float))
        }
        numeric_names = set.intersection(*(
            {name for name, value in row.items() if isinstance(value, (int, float))}
            for row in rows
        ))
        for name in sorted(numeric_names):
            values = [float(row[name]) for row in rows]
            record[name] = statistics.mean(values)
            record[f"{name}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        records.append(record)

    metadata = dict(payloads[0]["metadata"])
    metadata.update({
        "aggregation_type": "mean_and_sample_std_across_seeds",
        "seeds": args.seeds,
        "num_seeds": len(args.seeds),
    })
    metadata["arguments"] = dict(metadata.get("arguments", {}))
    metadata["arguments"]["seeds"] = args.seeds
    metadata["arguments"].pop("seed", None)
    output = {"metadata": metadata, "results": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    csv_path = args.output.with_suffix(".csv")
    columns = sorted({name for record in records for name in record})
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {args.output} and {csv_path}")


if __name__ == "__main__":
    main()
