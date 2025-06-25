"""
Modulo que define a classe AudioFile responsável por carregar,
armazenar e fornecer utilidades sobre um arquivo de áudio.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np
import librosa


class AudioFile:
    """Representa um arquivo de áudio único.

    Parameters
    ----------
    filepath : str | Path
        Caminho para o arquivo de áudio.
    sample_rate : int, optional
        Taxa de amostragem desejada (o arquivo será reamostrado se necessário),
        by default 16_000.
    """

    SUPPORTED_EXTS = {'.wav', '.flac', '.mp3', '.ogg'}

    def __init__(self, filepath: str | Path, sample_rate: int = 16_000) -> None:
        self.filepath = Path(filepath)
        if self.filepath.suffix.lower() not in self.SUPPORTED_EXTS:
            raise ValueError(f"Extensão {self.filepath.suffix} não suportada.")
        self.sample_rate = sample_rate
        self._signal: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Leitura e propriedades básicas
    # ------------------------------------------------------------------ #
    def load(self, mono: bool = True) -> np.ndarray:
        """Carrega o áudio em memória se ainda não foi carregado.

        Parameters
        ----------
        mono : bool, optional
            Converte para mono (média dos canais) se True, by default True.

        Returns
        -------
        np.ndarray
            Sinal de áudio como array 1‑D (ou 2‑D se mono=False).
        """
        if self._signal is None:
            self._signal, _ = librosa.load(
                self.filepath.as_posix(), sr=self.sample_rate, mono=mono
            )
        return self._signal

    @property
    def duration(self) -> float:
        """Duração em segundos."""
        self.load()
        return self._signal.shape[-1] / self.sample_rate  # type: ignore[index]

    def __repr__(self) -> str:  # pragma: no cover
        return f"AudioFile(path='{self.filepath}', sr={self.sample_rate})"
