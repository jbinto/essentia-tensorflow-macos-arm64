# tensorflow-metal with Essentia on macOS arm64 — model-dependent

> Assisted by Claude Code (Opus 4.7, 1M context), 2026-04-26.

**One-line recommendation:** `tensorflow-metal` is worth it for some Essentia
TF models and a net loss for others. On Apple M1 Pro, **transformer-/Inception-style
models (MAEST, VGGish) get a ~1.3× speedup**, while **EfficientNet-based
Discogs models lose 2.4–2.5×** and small CNN models (YAMNet, TempoCNN) lose
1.8–3.1×. Default off; opt in per model.

## Benchmark numbers

Hardware: Apple M1 Pro, 16 GB unified memory, macOS 26.1.
Audio: 97.41 s `Vivaldi_Sonata_5_II_Allegro.wav` (Essentia test corpus).
TensorFlow 2.15.0; `tensorflow-metal==1.2.0`.
1 warmup, 3 timed runs per model. Times are for `algorithm(audio)` only —
model construction and audio decode excluded.

| Model | Essentia algorithm | CPU median | GPU median | GPU/CPU | Verdict |
|---|---|---:|---:|---:|---|
| `discogs-effnet-bs64-1` | `TensorflowPredictEffnetDiscogs` | 799 ms | 1911 ms | **0.42×** | ❌ 2.4× slower |
| `discogs_track_embeddings-effnet-bs64-1` | `TensorflowPredictEffnetDiscogs` | 799 ms | 1995 ms | **0.40×** | ❌ 2.5× slower |
| `audioset-vggish-3` | `TensorflowPredictVGGish` | 1905 ms | 1486 ms | **1.28×** | ✅ 1.3× faster |
| `discogs-maest-30s-pw-2` | `TensorflowPredictMAEST` | 3815 ms | 2868 ms | **1.33×** | ✅ 1.3× faster |
| `audioset-yamnet-1` | `TensorflowPredictVGGish` | 316 ms | 582 ms | **0.54×** | ❌ 1.8× slower |
| `deepsquare-k16-3` | `TempoCNN` | 127 ms | 475 ms | **0.27×** | ❌ 3.7× slower |

GPU numbers are stable across two separate processes (re-run after a metal
re-install) — within-run variance is ~5–10 %, well below the 1.3× wins or
the 2× losses. Output is bit-stable across configurations.

(Skipped: the dense classifier heads — every `*-discogs-effnet-1.pb` and
`*-msd-musicnn-2.pb` in the model zoo. They're tiny `TensorflowPredict2D`
graphs that consume pre-computed embeddings, so end-to-end cost is dominated
by whichever embedder fed them.)

## When does Metal help?

Pattern: the wins are on **larger graphs with regular dense compute**
(MAEST is a transformer; VGGish is straight Conv2D + dense). The losses are
on **graphs with op-fallback boundaries or small per-call work** — EfficientNet
has many depthwise/grouped convolutions implemented via `SplitV`+`Concat`
patterns where Metal lacks a kernel and falls back to CPU, forcing host↔device
transfers in every block. TempoCNN and YAMNet are small enough that GPU
launch overhead per call dominates.

Per-op placement of `discogs-effnet-bs64` confirms this: 2076/2094 ops on
`GPU:0`, **18 ops on `CPU:0` — all `SplitV` + one `Const`**.

## Install

`tensorflow-metal` versions are pinned to TF minor releases. With
`tensorflow==2.15.0` (what this repo's recipe installs), the matching plugin
is `tensorflow-metal==1.2.0`. The version table is at
<https://developer.apple.com/metal/tensorflow-plugin/>.

```sh
source .venv/bin/activate
uv pip install tensorflow-metal==1.2.0
# verify TF sees the GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# -> [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

No relink needed. Essentia uses the TF C API via `libtensorflow_cc.2.dylib`
and creates its session with `TF_NewSessionOptions()` — no device pin — so
whatever PluggableDevice TF discovers at runtime is what gets used.

## Verifying the GPU is actually used

When you grep `TF_CPP_MIN_LOG_LEVEL=0` output for an Essentia run, you can
confirm three things:

1. **Metal plugin loaded:**
   `Metal device set to: Apple M1 Pro` and
   `Created TensorFlow device (/job:localhost/.../device:GPU:0 ...) -> physical PluggableDevice (... name: METAL ...)`.
2. **Grappler GPU optimizer ran on Essentia's frozen graph:**
   `Plugin optimizer for device_type GPU is enabled.`
3. **Per-op placement (with `TF_CPP_VMODULE=placer=2`):** for the
   Discogs-EffNet graph, 2076 ops on `GPU:0`, 18 on `CPU:0` (all `SplitV` +
   one `Const`).

So this isn't a "metal loads but isn't used" outcome — the GPU genuinely
runs the inference. Whether that's faster is the question above.

## Reproduction

Multi-model benchmark script: `~/src/essentia-build/bench_metal_all.py`.
Times **only** `algorithm(audio)` — model construction and audio decode are
excluded. 1 warmup, 3 timed runs, median + min/max.

For a clean CPU baseline, **uninstall** `tensorflow-metal` rather than trying
to hide the GPU from TF in-process. `tf.config.set_visible_devices([], 'GPU')`
only affects Python eager mode; Essentia's C-API session reads the global
SessionOptions, which still sees the Metal pluggable device. (Confirmed with
device-placement logging — when `set_visible_devices([], 'GPU')` is set in
Python and Essentia is imported afterwards, `Created TensorFlow device
(... PluggableDevice ... METAL ...)` still fires dozens of times during the
Essentia run, and the "CPU baseline" comes out indistinguishable from the
GPU run.)

```sh
# CPU baseline — bulletproof:
uv pip uninstall tensorflow-metal
python bench_metal_all.py vivaldi.wav models CPU

# GPU run:
uv pip install tensorflow-metal==1.2.0
python bench_metal_all.py vivaldi.wav models GPU
```

## Caveats

- Tested only on M1 Pro at each model's native batch/patch size. M2/M3/M4
  chips, larger GPUs (M-series Max/Ultra), or different audio lengths may
  shift the picture — especially for the EfficientNet losers, where larger
  GPU caches could absorb the SplitV-induced transfers.
- Re-test if you upgrade to `tensorflow>=2.16` — Apple has been tightening
  the plugin's op coverage, and the `SplitV` fallback may go away.
- Inference only. Says nothing about training (Essentia doesn't train anyway).
