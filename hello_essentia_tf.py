#!/usr/bin/env python3
"""Minimal hello-world: load an audio file, run Discogs-EffNet, print embedding shape.

Usage:
  python hello_essentia_tf.py <audio.wav> <discogs-effnet-bs64-1.pb>
"""
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quiet TF startup spam

import essentia
from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs


def main(audio_path: str, model_path: str) -> int:
    print(f"essentia: {essentia.__version__}")
    print(f"audio:    {audio_path}")
    print(f"model:    {model_path}")

    audio = MonoLoader(filename=audio_path, sampleRate=16000, resampleQuality=4)()
    print(f"audio:    {audio.shape[0]} samples @ 16 kHz "
          f"({audio.shape[0] / 16000:.2f} s)")

    embedder = TensorflowPredictEffnetDiscogs(
        graphFilename=model_path,
        output="PartitionedCall:1",  # 1280-dim embeddings tap
    )
    emb = embedder(audio)
    print(f"emb:      shape={emb.shape} dtype={emb.dtype} "
          f"mean={emb.mean():.4f} std={emb.std():.4f}")
    assert emb.ndim == 2 and emb.shape[1] == 1280, emb.shape
    print("OK")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
