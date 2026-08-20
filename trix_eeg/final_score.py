"""Final geometric aggregation for TriX-EEG."""

from __future__ import annotations

import math


def final_trix_score(
    ds: float,
    fs: float,
    rs: float,
    *,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """Implement equations (39)-(43) as a weighted geometric mean."""
    scores = (float(ds), float(fs), float(rs))
    weights = tuple(float(weight) for weight in weights)
    if any(not 0.0 <= score <= 1.0 for score in scores):
        raise ValueError(f"DS, FS, and RS must lie in [0,1], got {scores}")
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("Weights must be non-negative and have a positive sum")
    if any(score == 0.0 and weight > 0.0 for score, weight in zip(scores, weights)):
        return 0.0
    log_score = sum(
        weight * math.log(score)
        for score, weight in zip(scores, weights)
        if weight > 0.0
    ) / sum(weights)
    return math.exp(log_score)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Calculate the final TriX-EEG score")
    parser.add_argument("ds", type=float, help="Level-1 Denoise Score")
    parser.add_argument("fs", type=float, help="Level-2 Faithfulness Score")
    parser.add_argument("rs", type=float, help="Level-3 Reliability Score")
    parser.add_argument("--weights", nargs=3, type=float, default=(1.0, 1.0, 1.0))
    args = parser.parse_args()
    print(final_trix_score(args.ds, args.fs, args.rs, weights=tuple(args.weights)))


if __name__ == "__main__":
    main()
