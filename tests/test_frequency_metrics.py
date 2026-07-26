import unittest

import torch

from metrics import batch_frequency_preservation


class FrequencyPreservationMetricsTest(unittest.TestCase):
    def setUp(self):
        self.fs = 256.0
        t = torch.arange(512) / self.fs
        self.alpha_10hz = torch.sin(2 * torch.pi * 10.0 * t)[None, None]

    def test_identical_signal_is_perfectly_preserved(self):
        result = batch_frequency_preservation(
            self.alpha_10hz, self.alpha_10hz, fs=self.fs
        )
        self.assertAlmostEqual(result["psd_similarity"].item(), 1.0, places=6)
        self.assertAlmostEqual(result["alpha_peak_error_hz"].item(), 0.0)
        self.assertAlmostEqual(result["alpha_peak_preservation_rate"].item(), 1.0)
        for name, value in result.items():
            if name.startswith("band_power_absolute_error_"):
                self.assertAlmostEqual(value.item(), 0.0, places=7)

    def test_shifted_alpha_peak_is_detected(self):
        t = torch.arange(512) / self.fs
        alpha_12hz = torch.sin(2 * torch.pi * 12.0 * t)[None, None]
        result = batch_frequency_preservation(alpha_12hz, self.alpha_10hz, fs=self.fs)
        self.assertAlmostEqual(result["alpha_peak_error_hz"].item(), 2.0)
        self.assertAlmostEqual(result["alpha_peak_preservation_rate"].item(), 0.0)

    def test_amplitude_change_produces_absolute_band_error(self):
        result = batch_frequency_preservation(
            2.0 * self.alpha_10hz, self.alpha_10hz, fs=self.fs
        )
        self.assertGreater(result["band_power_absolute_error_alpha"].item(), 0.0)
        self.assertAlmostEqual(result["psd_similarity"].item(), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
