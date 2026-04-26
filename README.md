# essentia-tensorflow on macOS arm64 — a working build recipe

> Assisted by Claude Code (Opus 4.7, 1M context), 2026-04-26.

Getting `essentia` to import on Apple Silicon **with TensorFlow algorithms enabled**
is fiddly. The PyPI `essentia-tensorflow` arm64 wheel is broken
([upstream issue #1486](https://github.com/MTG/essentia/issues/1486)), and the
source build's own TF setup script is Linux-only. This repo documents what
actually works.

**Tested against** (last verified 2026-04-26):

| | |
|---|---|
| macOS | 26.1, arm64 (Apple Silicon) |
| Toolchain | Apple clang from Xcode CLT |
| Python | 3.11.15 |
| TensorFlow | 2.15.0 (from PyPI) |
| Essentia | `7e90d20b` (`v2.1_beta5-1418-g7e90d20b`, dated 2026-04-16) |

Essentia's `master` moves; the `git checkout` step below pins to the exact
commit this was tested against. If you re-test against a newer essentia,
please bump both the SHA and the date.

## tl;dr

> **Three things had to change to get this building:**
>
> 1. **Bump C++ standard to C++17.** Homebrew's Eigen is now 5.0.1, which
>    requires C++14+. Essentia defaults to C++11. Pass `--std=c++17` to
>    `./waf configure`.
> 2. **Replace the Linux TF setup script.** `src/3rdparty/tensorflow/setup_from_python.sh`
>    expects `lib*.so.2` and links `_pywrap_tensorflow_internal.so`. On macOS
>    arm64 the wheel ships `lib*.2.dylib`, and the C-API symbols live in
>    `libtensorflow_cc.2.dylib` (verified with `nm -gU` — `_pywrap_tensorflow_internal.so`
>    has zero `TF_*` exports on this platform). Use [`setup_tf_macos.sh`](./setup_tf_macos.sh).
> 3. **Install into the venv directly via `./waf install`.** The PyPI wheel's
>    Python submodule is broken; building from source and installing with
>    `--pythondir=$(python -c 'import site; print(site.getsitepackages()[0])')`
>    sidesteps it entirely.
>
> End result: `from essentia.standard import TensorflowPredictEffnetDiscogs`
> works, native arm64, ~1m25s build, no Bazel, no Rosetta.

## Setup

### Prerequisites

- Apple Silicon Mac, Xcode Command Line Tools (`xcode-select --install`)
- Homebrew packages:
  ```sh
  brew install eigen libyaml fftw ffmpeg libsamplerate taglib chromaprint pkgconf uv
  ```
  (`pkgconf` provides the `pkg-config` binary; the upstream `pkg-config` formula
  has been retired by Homebrew. `uv` is Astral's Python package/project manager
  — we use it to get a clean, fully-isolated Python 3.11 + TensorFlow 2.15
  environment without depending on whatever system Python you happen to have.)

### Build

```sh
# 1. Pick a working dir. Everything in this recipe lives under one directory.
mkdir -p ~/src/essentia-build && cd ~/src/essentia-build

# 2. Create an isolated Python 3.11 venv with TensorFlow 2.15 installed.
#    `uv` will download the correct CPython if you don't have it. The venv is
#    a regular `.venv/` — `source .venv/bin/activate` works the way you expect.
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install tensorflow==2.15.0    # CPU; provides the dylibs we link against

# 3. Get this repo (for the script + docs) and the essentia source.
#    Essentia is pinned to the exact commit this recipe was last verified
#    against — see the "Tested against" table above.
git clone https://github.com/jbinto/essentia-tensorflow-macos-arm64.git
git clone --recursive https://github.com/MTG/essentia.git
git -C essentia checkout 7e90d20b3965f334682930c9b02992c46d273717
git -C essentia submodule update --init --recursive
ln -s ../essentia essentia-tensorflow-macos-arm64/essentia       # convenience

# 4. Build the TF link context (creates ./tf-context/ with symlinks + pkg-config).
cd essentia-tensorflow-macos-arm64
PYTHON="$(cd .. && pwd)/.venv/bin/python" \
ESSENTIA_SRC="$(cd .. && pwd)/essentia" \
CTX_DIR="$(cd .. && pwd)/tf-context" \
  ./setup_tf_macos.sh

# 5. Configure and build essentia.
cd ../essentia
export PKG_CONFIG_PATH="$HOME/src/essentia-build/tf-context/lib/pkgconfig:$(brew --prefix eigen)/share/pkgconfig:$PKG_CONFIG_PATH"
PY_SITE=$(python -c "import site; print(site.getsitepackages()[0])")

./waf configure \
  --std=c++17 \
  --with-python \
  --with-tensorflow \
  --build-static \
  --prefix="$VIRTUAL_ENV" \
  --pythondir="$PY_SITE"

./waf -j$(sysctl -n hw.ncpu)
./waf install
```

If you see `Eigen requires at least c++14 support.`, you forgot `--std=c++17`.
If you see `Checking for 'tensorflow' : not found`, your `PKG_CONFIG_PATH` is
not pointing at the `tf-context/lib/pkgconfig` directory.

### Verify

```sh
python -c "from essentia.standard import TensorflowPredictEffnetDiscogs; print('ok')"
```

## Hello world (running a real model)

The TF algorithms in Essentia load a frozen `.pb` graph at construction time.
Models live in [Essentia's model zoo](https://essentia.upf.edu/models/). Each
model has a sibling `.json` describing its schema (input node, output nodes,
sample rate, expected pre-processing). For Discogs-EffNet, the embedding tap
is `PartitionedCall:1` (1280-dim) — see
[`models.json`](https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.json)
for the full schema.

```sh
# Grab the Discogs-EffNet graph (~18 MB) and its schema JSON.
mkdir -p ~/src/essentia-build/models
cd ~/src/essentia-build/models
curl -fLO --retry 3 https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb
curl -fLO --retry 3 https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.json

# Run the hello-world. Essentia's test corpus ships several wav files;
# `dubstep.wav` is short and works fine.
cd -
python hello_essentia_tf.py \
  ~/src/essentia-build/essentia/test/audio/recorded/dubstep.wav \
  ~/src/essentia-build/models/discogs-effnet-bs64-1.pb
```

To browse other models: `https://essentia.upf.edu/models/` is a plain Apache
directory listing (`feature-extractors/`, `classifiers/`, `pitch/`, `tempo/`,
etc.). Every `.pb` file has a sibling `.json` with the input/output schema and
recommended pre-processing — read that before swapping models in the script.

Expected output:

```
essentia: 2.1-beta6-dev
audio:    .../dubstep.wav
model:    .../discogs-effnet-bs64-1.pb
audio:    109601 samples @ 16 kHz (6.85 s)
emb:      shape=(5, 1280) dtype=float32 mean=0.0403 std=0.1170
OK
```

That's it — you have native-arm64 Essentia + TensorFlow inference working.

## Should I add `tensorflow-metal` for GPU acceleration?

It's model-dependent. On M1 Pro across six audio-input models in the Essentia
zoo, `tensorflow-metal==1.2.0` gives a **~1.3× speedup** on transformer/Inception
graphs (MAEST, VGGish) and a **2.4–3.7× slowdown** on EfficientNet-based
Discogs models, YAMNet, and TempoCNN — likely because `SplitV` ops fall back
to CPU and small graphs are dominated by per-call GPU launch overhead.
Default off; opt in per model. See [`METAL_NOTES.md`](./METAL_NOTES.md) for
the full table, install command, version pinning, op-fallback analysis, and
reproduction recipe.

## Technical writeup

### Why the PyPI wheel doesn't work

`pip install essentia-tensorflow` succeeds on macOS arm64 (`2.1b6.dev1389` as
of writing) but importing TF-using algorithms fails. The wheel was built but
something in the packaging step drops the TF surface. See
[issue #1486](https://github.com/MTG/essentia/issues/1486) for tracking.

### Why the upstream TF setup script doesn't work on macOS

`src/3rdparty/tensorflow/setup_from_python.sh` is hardcoded for the Linux
TensorFlow wheel layout, which differs from macOS in three ways:

| | Linux x86_64 | macOS arm64 |
|---|---|---|
| extension | `.so` | `.dylib` |
| versioned name | `libtensorflow_framework.so.2` | `libtensorflow_framework.2.dylib` |
| where `TF_*` symbols live | `_pywrap_tensorflow_internal.so` | `libtensorflow_cc.2.dylib` |
| linker hint | `LD_LIBRARY_PATH` / `ldconfig` | `@rpath/` install_name + `-Wl,-rpath,...` |

You can confirm the symbol claim yourself:

```sh
TF=$(python -c "import tensorflow,os; print(os.path.dirname(tensorflow.__file__))")
# 0 hits on macOS arm64:
nm -gU "$TF/python/_pywrap_tensorflow_internal.so" | grep -c "_TF_NewGraph"
# 309 hits on macOS arm64:
nm -gU "$TF/libtensorflow_cc.2.dylib" | grep -c " T _TF_"
```

So on macOS the C API consumer needs to link against `libtensorflow_cc` (and
`libtensorflow_framework` for transitive deps), not `_pywrap_tensorflow_internal`.

### What `setup_tf_macos.sh` does

It builds a small "link context" directory (`tf-context/`) that `pkg-config`
can advertise to waf:

```
tf-context/
├── include/tensorflow/c/        # copied from essentia/src/3rdparty/tensorflow/c
│   ├── c_api.h
│   └── tf_attrtype.h
└── lib/
    ├── libtensorflow_cc.dylib            -> libtensorflow_cc.2.dylib
    ├── libtensorflow_cc.2.dylib          -> $VENV/.../tensorflow/libtensorflow_cc.2.dylib
    ├── libtensorflow_framework.dylib     -> libtensorflow_framework.2.dylib
    ├── libtensorflow_framework.2.dylib   -> $VENV/.../tensorflow/libtensorflow_framework.2.dylib
    └── pkgconfig/tensorflow.pc
```

The two-tier symlink pattern matters because:

- The unversioned symlinks (`libtensorflow_cc.dylib`) let clang's `-ltensorflow_cc`
  flag resolve at link time. macOS's linker doesn't follow `.2.dylib` for `-l`.
- The versioned symlinks (`libtensorflow_cc.2.dylib`) preserve the install_name
  embedded in the dylib (`@rpath/libtensorflow_cc.2.dylib`), so the dynamic loader
  can resolve it at runtime.

The generated `tensorflow.pc`:

```
Libs: -L${libdir} -ltensorflow_cc -ltensorflow_framework
      -Wl,-rpath,${libdir} -Wl,-rpath,$TF_DIR
Cflags: -I${includedir}/tensorflow
```

…embeds **two** rpaths. Either alone is sufficient at runtime; both is robust.
(`${libdir}` resolves the @rpath via the symlink farm; `$TF_DIR` resolves it via
the actual venv install. If you ever blow away `tf-context/`, the second rpath
keeps things working.)

### Why C++17

Homebrew's Eigen pinned to 5.0.1 in early 2025. Eigen 5.x's
`<unsupported/Eigen/CXX11/Tensor>` (transitively pulled in by
`essentia/streaming/types.h`) requires C++14 or later. Essentia's `wscript`
defaults to C++11 via the `--std=` option; pass `--std=c++17` and the
errors-flood-of-template-noise go away. C++17 is also the standard TF 2.15
expects internally if you ever pull in C++ TF headers, so it's the safe choice.

### Why install into the venv via waf, not pip

The waf install step puts the compiled `_essentia.cpython-311-darwin.so` and
the Python wrappers into `$VIRTUAL_ENV/lib/python3.11/site-packages/essentia/`
directly. There's no pip metadata, but there's also no broken wheel — you just
get a working `import essentia.standard`. The cost is that the install isn't
visible to `pip list` and won't be re-resolved when you upgrade other packages;
you have to rebuild manually if Python or TensorFlow changes.

### A note on `essentia.tensorflow`

Issue #1486 talks about `import essentia.tensorflow` being missing. As of the
HEAD commit used here, `src/python/essentia/tensorflow/` does not exist in the
source tree at all — there's nothing to import as `essentia.tensorflow`. The TF
algorithms are exposed through `essentia.standard` (and `essentia.streaming`),
which is auto-generated from the C++ algorithm registry by the build. So the
real test of "does TF work" is:

```python
from essentia.standard import (
    TensorflowPredict,
    TensorflowPredictEffnetDiscogs,  # pick whichever
    TensorflowPredictMusiCNN,
    TensorflowPredictVGGish,
)
```

…and these all import cleanly with this build.

### Build performance

On an M-series Mac with `-j$(sysctl -n hw.ncpu)`:

- `./waf configure` — ~1 second
- `./waf` — **~1m25s** (340 compile units)
- `./waf install` — <1 second

No Bazel, no TensorFlow build from source. The TF dylibs are pre-built and
shipped with the `tensorflow` PyPI wheel; we just link against them.

## Re-running

```sh
cd ~/src/essentia-build/essentia && ./waf distclean
~/dev/essentia-tensorflow-macos-arm64/setup_tf_macos.sh   # if TF version or venv changed
# then re-run the configure + build steps
```

## License

The contents of this repo (the script and the writeup) are public domain — do
whatever. Essentia itself is AGPLv3; TensorFlow is Apache 2.0.
