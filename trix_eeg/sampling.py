"""Reproducible, stratified sampling for computationally expensive metrics."""

from __future__ import annotations

from collections import defaultdict
from numbers import Number

import torch
from torch.utils.data import Subset


def _scalar(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1)[0].item()
    if isinstance(value, Number):
        return value
    return str(value)


def _item_stratum(dataset, index: int):
    """Return the benchmark stratum without materializing data when possible."""
    if isinstance(dataset, Subset):
        return _item_stratum(dataset.dataset, int(dataset.indices[index]))

    samples = getattr(dataset, "samples", None)
    records = getattr(dataset, "records", None)
    if samples is not None and records is not None and samples:
        sample = samples[index]
        if isinstance(sample, tuple) and sample:
            record = records[int(sample[0])]
            if "subject_id" in record:
                return ("subject_id", _scalar(record["subject_id"]))

    if samples is not None and samples and isinstance(samples[index], dict):
        item = samples[index]
    else:
        item = dataset[index]
    for key in ("snr_db", "subject_id", "trial_id"):
        if key in item:
            return (key, _scalar(item[key]))
    return ("all", 0)


def stratified_subset_indices(dataset, max_samples: int, seed: int) -> list[int]:
    """Select a seeded, approximately balanced subset over SNR or trial strata.

    The same dataset, sample limit, and seed always produce the same indices.
    Different models can therefore share paired subsets simply by using the same
    seed. EEGDenoiseNet datasets are balanced over SNR and motion data over trial.
    """
    size = len(dataset)
    requested = size if max_samples <= 0 else min(size, int(max_samples))
    if requested >= size:
        return list(range(size))

    groups: dict[tuple, list[int]] = defaultdict(list)
    for index in range(size):
        groups[_item_stratum(dataset, index)].append(index)

    generator = torch.Generator().manual_seed(int(seed))
    labels = sorted(groups, key=str)
    label_order = torch.randperm(len(labels), generator=generator).tolist()
    base, remainder = divmod(requested, len(labels))
    quotas = {label: base for label in labels}
    for position in label_order[:remainder]:
        quotas[labels[position]] += 1

    # Redistribute any quota that a small stratum cannot satisfy.
    selected: list[int] = []
    unused: list[int] = []
    for label in labels:
        members = groups[label]
        order = torch.randperm(len(members), generator=generator).tolist()
        shuffled = [members[position] for position in order]
        take = min(quotas[label], len(shuffled))
        selected.extend(shuffled[:take])
        unused.extend(shuffled[take:])
    if len(selected) < requested:
        order = torch.randperm(len(unused), generator=generator).tolist()
        selected.extend(unused[position] for position in order[:requested - len(selected)])

    # Avoid presenting batches in stratum blocks.
    order = torch.randperm(len(selected), generator=generator).tolist()
    return [selected[position] for position in order]


def stratified_subset(dataset, max_samples: int, seed: int):
    indices = stratified_subset_indices(dataset, max_samples, seed)
    return dataset if len(indices) == len(dataset) else Subset(dataset, indices)
