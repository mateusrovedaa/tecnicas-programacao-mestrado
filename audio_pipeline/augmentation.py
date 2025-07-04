""" 
Augmentations e pipeline de dados de áudio.
"""

from __future__ import annotations

from typing import List, Protocol

import librosa
import numpy as np

class Transform(Protocol):
    """Interface para qualquer transformação (augmentação) de áudio."""

    def apply(self, signal: np.ndarray, sr: int) -> np.ndarray:  # noqa: D401
        ...

# -----------------------
# Transforms individuais
# -----------------------
class AddNoise:
    """Adiciona ruído branco gaussiano ao sinal."""

    def __init__(self, noise_level: float = 0.005) -> None:
        self.noise_level = noise_level

    def apply(self, signal: np.ndarray, sr: int) -> np.ndarray:  # noqa: D401
        noise = np.random.randn(len(signal)) * self.noise_level
        return signal + noise


class TimeStretch:
    """Estica ou comprime o sinal no tempo sem alterar o pitch."""

    def __init__(self, rate: float = 1.0) -> None:
        self.rate = rate

    def apply(self, signal: np.ndarray, sr: int) -> np.ndarray:
        return librosa.effects.time_stretch(signal, rate=self.rate)


class PitchShift:
    """Desloca o pitch em ``n_steps`` sem alterar a duração."""

    def __init__(self, n_steps: float = 2.0) -> None:
        self.n_steps = n_steps

    def apply(self, signal: np.ndarray, sr: int) -> np.ndarray:
        return librosa.effects.pitch_shift(signal, sr=sr, n_steps=self.n_steps)


# -------------------------
# Pipeline de augmentations
# -------------------------
class AugmentationPipeline:
    """Aplica uma sequência ordenada de ``Transform`` sobre o sinal."""

    def __init__(self, steps: List[Transform]):
        self.steps = steps

    def run(self, signal: np.ndarray, sr: int) -> np.ndarray:
        for step in self.steps:
            signal = step.apply(signal, sr)
        return signal
