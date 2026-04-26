#!/bin/bash
# Assisted by Claude Code (Opus 4.7, 1M context), 2026-04-26.
# Build a TensorFlow link context for Essentia on macOS arm64.
#
# The upstream src/3rdparty/tensorflow/setup_from_python.sh is Linux-only:
#   - it expects lib*.so.2 (macOS wheel ships lib*.2.dylib),
#   - it links against _pywrap_tensorflow_internal.so (which on macOS arm64
#     contains zero TF_* exports — confirm with `nm -gU`),
#   - it writes to /usr/local/ and runs ldconfig.
#
# On macOS arm64 the C-API symbols (TF_NewGraph, TF_LoadSessionFromSavedModel, ...)
# live in libtensorflow_cc.2.dylib. This script:
#   1. Symlinks libtensorflow_cc.2.dylib + libtensorflow_framework.2.dylib
#      into a local lib dir (versioned + unversioned aliases for `-l` resolution).
#   2. Copies the Essentia-bundled C-API headers in.
#   3. Writes a tensorflow.pc that waf will find via PKG_CONFIG_PATH.
#
# Usage:
#   PYTHON=/path/to/venv/bin/python \
#   ESSENTIA_SRC=/path/to/essentia \
#   CTX_DIR=/path/to/output/tf-context \
#     ./setup_tf_macos.sh
#
# Defaults: PYTHON=$(command -v python3), CTX_DIR=$PWD/tf-context, ESSENTIA_SRC=$PWD/essentia
set -euo pipefail

PYTHON="${PYTHON:-$(command -v python3)}"
CTX="${CTX_DIR:-$PWD/tf-context}"
ESSENTIA_SRC="${ESSENTIA_SRC:-$PWD/essentia}"

if [ ! -d "$ESSENTIA_SRC/src/3rdparty/tensorflow/c" ]; then
  echo "ERROR: ESSENTIA_SRC=$ESSENTIA_SRC does not look like an essentia checkout" >&2
  echo "       (missing src/3rdparty/tensorflow/c)" >&2
  exit 1
fi

TF_DIR=$("$PYTHON" -c "import tensorflow, os; print(os.path.dirname(tensorflow.__file__))")
TF_VERSION=$("$PYTHON" -c "import tensorflow; print(tensorflow.__version__)")

echo "PYTHON=$PYTHON"
echo "TF_DIR=$TF_DIR"
echo "TF_VERSION=$TF_VERSION"
echo "CTX=$CTX"

for f in libtensorflow_cc.2.dylib libtensorflow_framework.2.dylib; do
  if [ ! -f "$TF_DIR/$f" ]; then
    echo "ERROR: $TF_DIR/$f not found." >&2
    echo "       This script targets the macOS arm64 TensorFlow wheel layout." >&2
    exit 1
  fi
done

rm -rf "$CTX"
mkdir -p "$CTX/lib/pkgconfig" "$CTX/include/tensorflow"

# Symlink the dylibs. Keep the versioned name so the @rpath/lib*.2.dylib
# install_name embedded in each dylib resolves cleanly at runtime.
ln -sf "$TF_DIR/libtensorflow_cc.2.dylib"        "$CTX/lib/libtensorflow_cc.2.dylib"
ln -sf "$TF_DIR/libtensorflow_framework.2.dylib" "$CTX/lib/libtensorflow_framework.2.dylib"
# Unversioned aliases so clang's `-l` flag resolves them at link time.
ln -sf libtensorflow_cc.2.dylib        "$CTX/lib/libtensorflow_cc.dylib"
ln -sf libtensorflow_framework.2.dylib "$CTX/lib/libtensorflow_framework.dylib"

# Copy the C-API headers shipped with essentia.
rm -rf "$CTX/include/tensorflow/c"
cp -R "$ESSENTIA_SRC/src/3rdparty/tensorflow/c" "$CTX/include/tensorflow/c"

# Embed two rpaths: the local symlink farm (so the linker resolves the symlink,
# which retains the @rpath/ install_name) and the real venv tensorflow dir
# (so the @rpath/ install_name resolves at runtime). Either alone would work.
cat > "$CTX/lib/pkgconfig/tensorflow.pc" <<EOF
prefix=$CTX
libdir=\${prefix}/lib
includedir=\${prefix}/include

Name: tensorflow
Description: TensorFlow C API (linked against tensorflow Python wheel)
Version: $TF_VERSION
Libs: -L\${libdir} -ltensorflow_cc -ltensorflow_framework -Wl,-rpath,\${libdir} -Wl,-rpath,$TF_DIR
Cflags: -I\${includedir}/tensorflow
EOF

echo
echo "=== wrote $CTX/lib/pkgconfig/tensorflow.pc ==="
cat "$CTX/lib/pkgconfig/tensorflow.pc"
echo
echo "now: export PKG_CONFIG_PATH=\"$CTX/lib/pkgconfig:\$PKG_CONFIG_PATH\""
