"""
Augmentations simples para aumentar robustez do modelo.
"""

from __future__ import annotations

from typing import List

import librosa
import numpy as np


class AudioAugmenter:
    """Classe‑base de augmentations."""

    def apply(self, signal: np.ndarray, sr: int) -> np.ndarray:
        raise NotImplementedError


class AddNoise(AudioAugmenter):
    """Adiciona ruído branco Gaussian ao sinal."""

    def __init__(self, noise_level: float = 0.005) -> None:
        self.noise_level = noise_level

    def apply(self, signal: np.ndarray, sr: int) -> np.ndarray:
        noise = np.random.randn(len(signal)) * self.noise_level
        return signal + noise


class TimeStretch(AudioAugmenter):
    """Estica/comprime tempo sem alterar pitch."""

    def __init__(self, rate: float = 1.0) -> None:
        self.rate = rate

    def apply(self, signal: np.ndarray, sr: int) -> np.ndarray:
        return librosa.effects.time_stretch(signal, rate=self.rate)


class PitchShift(AudioAugmenter):
    """Altera o pitch em `n_steps` sem mudar a duração."""
    def __init__(self, n_steps: float = 2.0) -> None:
        self.n_steps = n_steps

    def apply(self, signal: np.ndarray, sr: int) -> np.ndarray:
        return librosa.effects.pitch_shift(signal, sr=sr, n_steps=self.n_steps)


class AugmentationPipeline:
    """Aplica augmentations em sequência."""

    def __init__(self, augmenters: List[AudioAugmenter]) -> None:
        self.augmenters = augmenters

    def run(self, signal: np.ndarray, sr: int) -> np.ndarray:
        for aug in self.augmenters:
            signal = aug.apply(signal, sr)
        return signal
