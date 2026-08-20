from types import SimpleNamespace

import pytest
import torch

from evaluate_custom_denoiser import ModelAdapter, _extract_state_dict, build_parser
from evaluate_trix_eeg import _experiments, validate_cli_args


class TwoDimensionalDenoiser(torch.nn.Module):
    def forward(self, inputs):
        return 0.5 * inputs


def test_custom_adapter_supports_bt_tensor_models_and_gradients():
    model = ModelAdapter(TwoDimensionalDenoiser(), input_layout="bt")
    inputs = torch.randn(2, 1, 32, requires_grad=True)
    prediction = model(inputs)["y_hat"]
    assert prediction.shape == inputs.shape
    prediction.sum().backward()
    assert inputs.grad is not None


def test_state_dict_extraction_accepts_standard_keys_and_dataparallel_prefix():
    tensor = torch.ones(1)
    state = _extract_state_dict({"model_state_dict": {"module.weight": tensor}}, None)
    assert list(state) == ["weight"]
    assert state["weight"] is tensor


def test_custom_parser_requires_weight_file_and_model_name():
    args = build_parser().parse_args([
        "--weight-file", "weights.pt", "--model-name", "external",
    ])
    assert args.weight_file == "weights.pt"
    assert args.model_factory is None


def test_builtin_selection_yields_only_requested_denoiser():
    args = SimpleNamespace(
        denoiser="fcnn",
        benchmark_checkpoint_root="benchmarks",
        braanet_checkpoint_root="braanet",
        motion_checkpoint_root="motion",
    )
    experiments = list(_experiments("eeg_eog", args))
    assert experiments == [("benchmark", "fcnn", "benchmarks/fcnn_eog/best.pt")]


def test_checkpoint_override_requires_one_model_and_dataset():
    with pytest.raises(ValueError, match="requires one"):
        validate_cli_args(SimpleNamespace(
            checkpoint="best.pt", denoiser="all", datasets=["eeg_eog"]
        ))
