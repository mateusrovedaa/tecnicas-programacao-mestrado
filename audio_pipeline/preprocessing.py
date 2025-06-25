"""
Transformações e pipeline de pré‑processamento de áudio.
"""

from __future__ import annotations

from typing import List, Protocol

import librosa
import numpy as np


class Transform(Protocol):
    """Interface para qualquer transformação de sinal."""

    def apply(self, signal: np.ndarray, sr: int) -> np.ndarray: ...


# ----------------------- #
# Transforms individuais  #
# ----------------------- #
class SilenceTrimmer:
    """Remove silêncio usando o método ``librosa.effects.trim``."""

    def __init__(self, top_db: float = 20.0) -> None:
        self.top_db = top_db

    def apply(self, signal: np.ndarray, sr: int) -> np.ndarray:
        trimmed, _ = librosa.effects.trim(signal, top_db=self.top_db)
        return trimmed


class Normalizer:
    """Normaliza amplitude para o intervalo [-1, 1]."""

    def apply(self, signal: np.ndarray, sr: int) -> np.ndarray:
        peak = np.max(np.abs(signal))
        return signal if peak == 0 else signal / peak


# ----------------------------- #
# Pipeline de pré‑processamento #
# ----------------------------- #
class PreprocessingPipeline:
    """Aplica uma sequência ordenada de ``Transform`` sobre o sinal."""

    def __init__(self, steps: List[Transform]) -> None:
        self.steps = steps

    def run(self, signal: np.ndarray, sr: int) -> np.ndarray:
        for step in self.steps:
            signal = step.apply(signal, sr)
        return signal
