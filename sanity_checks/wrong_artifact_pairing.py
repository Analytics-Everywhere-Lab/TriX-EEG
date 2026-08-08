#!/usr/bin/env python3
"""Recompute Level-2 metrics using artifacts drawn from another artifact type."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from sanity_checks.common import (
    DATASET_CHOICES,
    MetricAccumulator,
    add_common_arguments,
    add_metric_arguments,
    base_metadata,
    build_dataset,
    load_model,
    make_loader,
    metric_record,
    select_device,
    set_seed,
    signals_from_batch,
    write_results,
)
from trix_eeg.common import as_bct, predict
from trix_eeg.level2_removal_faithfulness import level2_removal_faithfulness


def artifact_from_item(item):
    value = item.get("artifact", item.get("noise"))
    return item["x"] - item["clean"] if value is None else value


class WrongArtifactDataset(Dataset):
    """Attach a deterministic random donor artifact to every target sample."""

    def __init__(self, target, donor, seed: int):
        if not len(donor):
            raise ValueError("Donor dataset is empty")
        self.target = target
        self.donor = donor
        generator = torch.Generator().manual_seed(int(seed))
        self.order = torch.randperm(len(donor), generator=generator).tolist()

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        target_item = dict(self.target[index])
        donor_index = self.order[index % len(self.order)]
        wrong = artifact_from_item(self.donor[donor_index]).float()
        if wrong.shape != target_item["x"].shape:
            raise ValueError(
                f"Target shape {tuple(target_item['x'].shape)} and donor artifact "
                f"shape {tuple(wrong.shape)} differ"
            )
        target_item["wrong_artifact"] = wrong
        return target_item


def rms_match(wrong: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    dimensions = tuple(range(1, wrong.ndim))
    wrong_rms = wrong.square().mean(dim=dimensions, keepdim=True).sqrt()
    true_rms = true.square().mean(dim=dimensions, keepdim=True).sqrt()
    return wrong * (true_rms / wrong_rms.clamp_min(1e-8))


def main(args) -> None:
    if args.donor_dataset == args.dataset:
        raise ValueError("--donor-dataset must be a different artifact type")
    set_seed(args.seed)
    device = select_device(args.cpu)
    paired = WrongArtifactDataset(
        build_dataset(args), build_dataset(args, args.donor_dataset), args.seed
    )
    loader = make_loader(paired, args)
    model, model_metadata = load_model(args, device)
    accumulators = {
        "true_artifact": MetricAccumulator(),
        "wrong_artifact_raw": MetricAccumulator(),
        "wrong_artifact_rms_matched": MetricAccumulator(),
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc="Wrong-artifact pairing"):
            noisy, clean, artifact = signals_from_batch(batch, device)
            prediction = predict(model, noisy)
            wrong = as_bct(batch["wrong_artifact"].to(device).float())
            references = {
                "true_artifact": artifact,
                "wrong_artifact_raw": wrong,
                "wrong_artifact_rms_matched": rms_match(wrong, artifact),
            }
            for condition, reference in references.items():
                metrics = level2_removal_faithfulness(
                    noisy, prediction, clean, reference,
                    raer_error_max=args.raer_error_max,
                    are_error_max=args.are_error_max,
                    residual_ratio_max=args.residual_ratio_max,
                )
                accumulators[condition].update({
                    name: metrics[name] for name in (
                        "ras", "raer", "utility_raer",
                        "artifact_reconstruction_error", "utility_are",
                        "rar", "utility_rar", "fs",
                    )
                })

    records = [
        metric_record(condition, accumulator.means(), donor_dataset=args.donor_dataset)
        for condition, accumulator in accumulators.items()
    ]
    true_metrics = accumulators["true_artifact"].means()
    for condition in ("wrong_artifact_raw", "wrong_artifact_rms_matched"):
        values = accumulators[condition].means()
        records.append(metric_record(
            f"{condition}_minus_true",
            {name: values[name] - true_metrics[name] for name in true_metrics},
            donor_dataset=args.donor_dataset,
        ))
    metadata = base_metadata(args, model_metadata)
    metadata["expected_behavior"] = {
        "ras": "lower for wrong artifacts",
        "raer": "may change for raw donors; RMS-matching isolates morphology from energy scale",
        "artifact_reconstruction_error": "higher for wrong artifacts",
        "rar": "changes because its projection uses the artifact reference",
        "fs": "lower for wrong artifacts, including the ARE penalty",
        "omitted": "NLR does not consume the artifact reference",
    }
    metadata["donor_dataset"] = args.donor_dataset
    write_results(args.output, records, metadata)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    add_metric_arguments(parser)
    parser.add_argument("--donor-dataset", choices=DATASET_CHOICES, default="eeg_emg")
    parser.add_argument("--output", default="runs/sanity_checks/wrong_artifact_pairing.json")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
