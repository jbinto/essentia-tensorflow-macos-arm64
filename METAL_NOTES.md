# tensorflow-metal with Essentia on macOS arm64 — don't bother

> Assisted by Claude Code (Opus 4.7, 1M context), 2026-04-26.

**One-line recommendation:** on Apple Silicon, do not install `tensorflow-metal`
for use with Essentia's `TensorflowPredict*` algorithms. It installs cleanly,
the Metal GPU is detected, ops actually do land on `GPU:0` — and the result
is **~2.1× slower** than plain CPU TensorFlow.

## TL;DR numbers

| | TF only (CPU) | TF + tensorflow-metal 1.2.0 (GPU) |
|---|---:|---:|
| 97.41 s Vivaldi → `(97, 1280)` embedding (Discogs-EffNet) | **883 ms** median | 1810–1904 ms median |
| Range across 3 timed runs | 807–1034 ms | 1806–2041 ms |
| Speedup | 1.00× | **0.46–0.49×** (i.e. slower) |

Hardware: Apple M1 Pro, 16 GB unified memory, macOS 26.1.

The two GPU columns are from two separate processes (re-run after a metal
re-install) — within-run variance is ~5–10 %, well below the 2× CPU/GPU gap.
Output is bit-stable across configurations (`mean=0.0668` either way).

## Install

`tensorflow-metal` versions are pinned to TF minor releases. With
`tensorflow==2.15.0` (what this repo's recipe installs), the matching plugin is
`tensorflow-metal==1.2.0`. The version table is at
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
   Discogs-EffNet graph, **2076 ops on GPU:0, 18 ops on CPU:0** (the 18 are
   `SplitV` calls + one `Const` — see "op fallback" below). All `Conv2D`,
   `MatMul`, `BiasAdd`, `Mean`, the wrapping `PartitionedCall`/
   `StatefulPartitionedCall` are on GPU:0. So this isn't a "metal loads but
   isn't used" outcome — the GPU genuinely runs the inference, it's just slow.

## Why it's slower

Best guess: the 18 `SplitV` ops fall back to CPU (~0.9 % of ops by count,
but they sit in the middle of every depthwise/grouped-conv block in
EfficientNet). Each fallback forces a host↔device tensor transfer, and at
this model size and audio length (`97 × 64 × 96 × 96` patches) the per-call
overhead dominates whatever the GPU saves on Conv2D throughput. The M1 Pro's
16-core GPU is ~2 TFLOPS fp32; the same chip's 8 CPU cores via Accelerate
(BLAS) and TF's MKLDNN-equivalent kernels easily compete with that for batch
sizes this small.

The Apple Metal pluggable-device documentation is honest about this:
"performance may vary depending on the model and operations used." For
small-batch CNN inference with many `SplitV`/concat boundaries, "vary" means
"lose by 2×."

## Reproduction

The benchmark script and CPU/GPU runs are in `~/src/essentia-build/bench_metal.py`.
Times **only** `embedder(audio)` — model construction and audio decode are
excluded. 1 warmup run, 3 timed runs, median + min/max.

For a clean CPU baseline, **uninstall** `tensorflow-metal` rather than trying
to hide the GPU from TF in-process. `tf.config.set_visible_devices([], 'GPU')`
only affects Python eager mode; Essentia's C-API session reads the global
SessionOptions, which still sees the Metal pluggable device. (We confirmed
this with device-placement logging — when `set_visible_devices([], 'GPU')` is
set in Python and Essentia is imported afterwards, `Created TensorFlow device
(... PluggableDevice ... METAL ...)` still fires dozens of times during the
Essentia run, and CPU-baseline timings come out indistinguishable from the
GPU timings.)

```sh
# CPU baseline — bulletproof:
uv pip uninstall tensorflow-metal
python bench_metal.py vivaldi.wav discogs-effnet-bs64-1.pb

# GPU run:
uv pip install tensorflow-metal==1.2.0
python bench_metal.py vivaldi.wav discogs-effnet-bs64-1.pb
```

## Caveats

- Tested only on M1 Pro with Discogs-EffNet at the model's native batch size
  (`bs64`). M2/M3/M4 chips, larger GPUs (M-series Max/Ultra), or different
  models (MusiCNN, VGGish, larger transformer-style audio models) may flip
  the result. The "don't bother" recommendation is for this specific
  configuration.
- Re-test if you upgrade to `tensorflow>=2.16` — Apple has been tightening
  the plugin's op coverage, and the `SplitV` fallback may go away.
- This says nothing about training. We tested inference only.
