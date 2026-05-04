"""
SpecAugment for MFCC features.
Applies frequency masking and time masking.
"""

import torch


class SpecAugment:
    """SpecAugment: frequency masking + time masking on MFCC features."""

    def __init__(
        self,
        freq_mask_param=10,
        time_mask_param=40,
        num_freq_masks=2,
        num_time_masks=2,
    ):
        """
        Args:
            freq_mask_param: Maximum width of frequency masks (MFCC bins).
            time_mask_param: Maximum width of time masks (time steps).
            num_freq_masks: Number of frequency masks to apply per sample.
            num_time_masks: Number of time masks to apply per sample.
        """
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def _freq_mask(self, features):
        n_channels, n_mfcc, _ = features.shape
        for _ in range(self.num_freq_masks):
            f = torch.randint(0, self.freq_mask_param + 1, (1,)).item()
            if f == 0:
                continue
            f0 = torch.randint(0, n_mfcc - f + 1, (1,)).item()
            features[:, f0 : f0 + f, :] = 0
        return features

    def _time_mask(self, features):
        n_channels, _, time_steps = features.shape
        for _ in range(self.num_time_masks):
            t = torch.randint(0, self.time_mask_param + 1, (1,)).item()
            if t == 0:
                continue
            t0 = torch.randint(0, time_steps - t + 1, (1,)).item()
            features[:, :, t0 : t0 + t] = 0
        return features

    def __call__(self, features):
        features = self._freq_mask(features)
        features = self._time_mask(features)
        return features
