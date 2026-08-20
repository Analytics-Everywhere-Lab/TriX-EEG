#!/usr/bin/env python3
"""Evaluate an external EEG denoiser with the standard TriX-EEG protocol."""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Mapping

import numpy as np
import torch

from evaluate_trix_eeg import (
    _eeg_dataset,
    _loader,
    _motion_dataset,
    _write_outputs,
    build_parser as build_protocol_parser,
    print_evaluation_profile,
)
from trix_eeg.evaluator import TriXConfig, evaluate_trix_model, evaluate_trix_model_multiseed


class ModelAdapter(torch.nn.Module):
    """Normalize common third-party input/output conventions for TriX-EEG."""

    def __init__(self, model, input_layout="bct", output_key="y_hat"):
        super().__init__()
        self.model = model
        self.input_layout = input_layout
        self.output_key = output_key

    def forward(self, inputs):
        model_inputs = inputs.squeeze(1) if self.input_layout == "bt" else inputs
        output = self.model(model_inputs)
        if isinstance(output, Mapping):
            if self.output_key not in output:
                raise KeyError(
                    f"Model output has keys {list(output)}; expected {self.output_key!r}"
                )
            output = output[self.output_key]
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not torch.is_tensor(output):
            raise TypeError(f"Model must return a tensor, mapping, or tuple; got {type(output)}")
        if output.ndim == 2:
            output = output.unsqueeze(1)
        return {"y_hat": output}


def _load_json_object(value):
    if not value:
        return {}
    if os.path.isfile(value):
        with open(value) as stream:
            value = stream.read()
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--model-kwargs must be a JSON object or a path to one")
    return parsed


def _import_factory(spec):
    if ":" not in spec:
        raise ValueError("--model-factory must use the form package.module:function")
    module_name, function_name = spec.rsplit(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    if not callable(factory):
        raise TypeError(f"Model factory {spec!r} is not callable")
    return factory


def _extract_state_dict(payload, state_dict_key):
    if state_dict_key:
        if not isinstance(payload, Mapping) or state_dict_key not in payload:
            raise KeyError(f"Checkpoint has no {state_dict_key!r} state-dict key")
        state_dict = payload[state_dict_key]
    elif isinstance(payload, Mapping) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    elif isinstance(payload, Mapping) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload
    if not isinstance(state_dict, Mapping) or not all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in state_dict.items()
    ):
        raise TypeError("Weight file does not contain a tensor state dict")
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def load_external_model(args, device):
    """Load TorchScript directly, or construct a model and apply a state dict."""
    if args.model_factory is None:
        try:
            return torch.jit.load(args.weight_file, map_location=device), {"format": "torchscript"}
        except (RuntimeError, ValueError) as error:
            raise ValueError(
                "The weight file is not TorchScript. Provide --model-factory "
                "package.module:function to construct its architecture."
            ) from error

    model = _import_factory(args.model_factory)(**_load_json_object(args.model_kwargs))
    if not isinstance(model, torch.nn.Module):
        raise TypeError("The model factory must return torch.nn.Module")
    try:
        payload = torch.load(args.weight_file, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(args.weight_file, map_location="cpu")
    state_dict = _extract_state_dict(payload, args.state_dict_key)
    incompatibility = model.load_state_dict(state_dict, strict=not args.non_strict)
    metadata = {
        "format": "state_dict",
        "factory": args.model_factory,
        "missing_keys": list(incompatibility.missing_keys),
        "unexpected_keys": list(incompatibility.unexpected_keys),
    }
    if isinstance(payload, Mapping):
        metadata["checkpoint_epoch"] = payload.get("epoch")
    return model.to(device), metadata


def _config(args):
    return TriXConfig(
        aggregation=args.aggregation,
        fs=args.fs,
        delta_snr_max=args.delta_snr_max,
        band_power_error_max=args.band_power_error_max,
        raer_error_max=args.raer_error_max,
        are_error_max=args.are_error_max,
        residual_ratio_max=args.residual_ratio_max,
        taas_frequency_range=(args.taas_frequency_low, args.taas_frequency_high),
        taas_n_fft=args.taas_n_fft,
        taas_hop_length=args.taas_hop_length,
        deletion_fractions=tuple(args.deletion_fractions),
        random_deletion_repeats=args.random_deletion_repeats,
        stability_noise_fraction=args.stability_noise_fraction,
        level3_max_samples=args.level3_max_samples,
        level3_batch_size=args.level3_batch_size,
        seed=args.seed,
        weights=tuple(args.weights),
    )


def main(args):
    if not os.path.isfile(args.weight_file):
        raise FileNotFoundError(f"Weight file does not exist: {args.weight_file}")
    if args.taas_n_fft <= 1 or args.taas_n_fft > args.signal_length:
        raise ValueError("--taas-n-fft must be in [2, signal length]")
    if args.taas_hop_length <= 0:
        raise ValueError("--taas-hop-length must be positive")
    if not 0.0 <= args.taas_frequency_low < args.taas_frequency_high <= args.fs / 2.0:
        raise ValueError("TAAS frequencies must satisfy 0 <= low < high <= fs/2")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    raw_model, load_metadata = load_external_model(args, device)
    model = ModelAdapter(raw_model, args.input_layout, args.output_key).to(device).eval()
    config = _config(args)
    results = []

    for dataset_name in args.datasets:
        if dataset_name == "eeg_eog":
            dataset, dataset_metadata = _eeg_dataset("EOG", args), {}
        elif dataset_name == "eeg_emg":
            dataset, dataset_metadata = _eeg_dataset("EMG", args), {}
        else:
            dataset, test_ids = _motion_dataset(args)
            dataset_metadata = {"test_subjects": test_ids}
        loader = _loader(dataset, args, device)
        description = f"{dataset_name}/{args.model_name}"
        if args.evaluation_seeds:
            evaluation, evaluation_std = evaluate_trix_model_multiseed(
                model, loader, device, config, args.evaluation_seeds,
                description=description,
            )
        else:
            evaluation = evaluate_trix_model(
                model, loader, device, config, description=description
            )
            evaluation_std = None
        result = {
            "dataset": dataset_name,
            "family": "external",
            "model": args.model_name,
            "checkpoint": args.weight_file,
            "checkpoint_epoch": load_metadata.get("checkpoint_epoch"),
            **dataset_metadata,
            "evaluation": evaluation,
        }
        if evaluation_std is not None:
            result["evaluation_std"] = evaluation_std
        results.append(result)
        print(f"\n{description}")
        print_evaluation_profile(evaluation)

    metadata = {
        "device": str(device),
        "datasets": args.datasets,
        "snr_levels": args.snr_levels,
        "framework_settings": config.__dict__,
        "evaluation_seeds": args.evaluation_seeds,
        "external_model": load_metadata,
        "failures": [],
    }
    _write_outputs(results, args.output_json, args.output_csv, metadata)
    print(f"Saved {len(results)} evaluations to {args.output_json} and {args.output_csv}")


def build_parser():
    parser = build_protocol_parser()
    parser.description = __doc__
    for action in list(parser._actions):
        if action.dest in {
            "denoiser", "checkpoint", "benchmark_checkpoint_root",
            "braanet_checkpoint_root", "motion_checkpoint_root", "skip_failed",
        }:
            for option in action.option_strings:
                parser._option_string_actions.pop(option, None)
            parser._actions.remove(action)
            for group in parser._action_groups:
                if action in group._group_actions:
                    group._group_actions.remove(action)
    parser.add_argument(
        "--weight-file", required=True, help="TorchScript or state-dict file"
    )
    parser.add_argument("--model-name", required=True, help="Name recorded in reports")
    parser.add_argument(
        "--model-factory", default=None,
        help="For a state dict: import path package.module:function returning nn.Module",
    )
    parser.add_argument(
        "--model-kwargs", default="{}",
        help="JSON object (or JSON file) passed to the model factory",
    )
    parser.add_argument("--state-dict-key", default=None)
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument("--input-layout", choices=["bct", "bt"], default="bct")
    parser.add_argument("--output-key", default="y_hat")
    parser.set_defaults(
        output_json="runs/trix_eeg/custom_results.json",
        output_csv="runs/trix_eeg/custom_summary.csv",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
