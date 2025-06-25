"""
Importa símbolos úteis no namespace do pacote para facilitar o uso
"""

from .audio_file import AudioFile
from .preprocessing import PreprocessingPipeline, SilenceTrimmer, Normalizer
from .augmentation import (
    AugmentationPipeline,
    AddNoise,
    TimeStretch,
    PitchShift,
)
from .dataset_loader import DatasetLoader

__all__ = [
    'AudioFile',
    'DatasetLoader',
    'PreprocessingPipeline',
    'SilenceTrimmer',
    'Normalizer',
    'AugmentationPipeline',
    'AddNoise',
    'TimeStretch',
    'PitchShift',
]
