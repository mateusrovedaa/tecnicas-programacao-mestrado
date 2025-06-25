"""
Fluxo completo: lê metadados + áudios, processa
todos os arquivos e grava a saída em .npy.

Para executar:
$ pip install requirements.txt
$ python main.py
"""
from pathlib import Path
import numpy as np
from tqdm import tqdm  # barra de progresso (pip install tqdm)

from audio_pipeline import (
    DatasetLoader,
    PreprocessingPipeline,
    SilenceTrimmer,
    Normalizer,
    AugmentationPipeline,
    AddNoise,
    TimeStretch,
    PitchShift,
    AudioFile,
)


# --------------------------------------------------------------------------- #
# Fábrica de pipelines – mantém construção isolada e reaproveitável           #
# --------------------------------------------------------------------------- #
def build_pipelines():
    """Retorna (preprocess, augment, normalizer_final)."""
    preprocess = PreprocessingPipeline(
        [SilenceTrimmer(top_db=20.0), Normalizer()]
    )
    augment = AugmentationPipeline(
        [AddNoise(0.01), TimeStretch(1.05), PitchShift(1.0)]
    )
    norm_final = Normalizer()
    return preprocess, augment, norm_final


# --------------------------------------------------------------------------- #
# Processamento de uma partição (train/val/test)                              #
# --------------------------------------------------------------------------- #
def process_partition(
    name: str,
    partition: list[dict],
    preprocess: PreprocessingPipeline,
    augment: AugmentationPipeline,
    normalizer: Normalizer,
    out_root: Path,
):
    """Aplica as etapas em todos os arquivos da partição e grava .npy."""
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = []
    for rec in tqdm(partition, desc=f"→ {name}", unit="file"):
        af = AudioFile(rec["filepath"])
        sig = preprocess.run(af.load(), af.sample_rate)
        sig = augment.run(sig, af.sample_rate)
        sig = normalizer.apply(sig, af.sample_rate)

        # salva em float32 (padrão para Deep Learning)
        np.save(out_dir / (af.filepath.stem + ".npy"), sig.astype(np.float32))
        sizes.append(sig.shape[0])

    if sizes:
        print(
            f"{name}: {len(sizes)} arquivos • duração média "
            f"{np.mean(sizes)/af.sample_rate:,.2f} s"
        )
    else:
        print(f"{name}: partição vazia.")


# --------------------------------------------------------------------------- #
# Script principal                                                            #
# --------------------------------------------------------------------------- #
def main():
    base_dir = Path(__file__).parent
    data_dir = base_dir / "sample_data" / "audio"
    meta_csv = base_dir / "sample_data" / "metadata.csv"
    out_root = base_dir / "processed"

    # 1) Carregar metadados
    loader = DatasetLoader(data_dir, meta_csv)
    loader.load_metadata()
    train, val, test = loader.split(ratios=(0.67, 0.17, 0.16))  # força ≥1 em val

    print(
        f"Samples → train={len(train)}, val={len(val)}, test={len(test)}"
    )

    # 2) Instanciar pipelines
    preprocess, augment, norm_final = build_pipelines()

    # 3) Processar cada partição
    for name, part in [("train", train), ("val", val), ("test", test)]:
        process_partition(
            name, part, preprocess, augment, norm_final, out_root
        )


if __name__ == "__main__":
    main()
