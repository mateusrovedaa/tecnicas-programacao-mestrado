"""
Carrega arquivos de áudio e seus metadados CSV.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import List, Tuple, Dict

from audio_pipeline.audio_file import AudioFile


class DatasetLoader:
    """Responsável por descobrir arquivos e juntá-los com os metadados."""

    def __init__(self, data_dir: str | Path, metadata_csv: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.metadata_csv = Path(metadata_csv)
        self.records: List[Dict] = []

    # ----------------------------------------- #
    # Métodos públicos                           #
    # ----------------------------------------- #
    def load_metadata(self) -> None:
        """Lê CSV e popula ``self.records`` com dicts contendo
        ``filepath`` e todos os campos do CSV (label, pig_id, etc.)."""
        with self.metadata_csv.open(newline='', encoding='utf‑8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                wav_path = self.data_dir / row['filename']
                if wav_path.exists():
                    row['filepath'] = wav_path
                    self.records.append(row)

    def split(
        self,
        ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int | None = 42,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Divide ``self.records`` em train/val/test mantendo shuffle."""
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.records)
        n = len(self.records)
        t = int(ratios[0] * n)
        v = int((ratios[0] + ratios[1]) * n)
        return self.records[:t], self.records[t:v], self.records[v:]

    # ---------------------------- #
    # Conveniência                 #
    # ---------------------------- #
    def iter_audiofiles(self):
        for r in self.records:
            yield AudioFile(r['filepath'])
