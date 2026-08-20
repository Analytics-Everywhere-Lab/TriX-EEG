"""TriX-EEG trustworthy artifact-removal evaluation framework."""

from .final_score import final_trix_score
from .level1_signal_validity import level1_signal_validity
from .level2_removal_faithfulness import level2_removal_faithfulness
from .level3_explanation_reliability import (
    gradient_attribution,
    gradient_x_input_attribution,
    integrated_gradients_attribution,
    level3_explanation_reliability,
    smoothgrad_attribution,
    temporal_artifact_alignment,
)

__all__ = [
    "level1_signal_validity",
    "level2_removal_faithfulness",
    "level3_explanation_reliability",
    "gradient_attribution",
    "gradient_x_input_attribution",
    "smoothgrad_attribution",
    "integrated_gradients_attribution",
    "temporal_artifact_alignment",
    "final_trix_score",
]
