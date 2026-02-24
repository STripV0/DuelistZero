#!/bin/bash
# Build script for ygopro-core -> libocgcore.so
# Prerequisites: g++, Lua 5.3 headers and lib
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
CORE_DIR="$PROJECT_ROOT/vendor/ygopro-core"
LUA_DIR="$PROJECT_ROOT/vendor/lua-5.3.5"
OUT_DIR="$PROJECT_ROOT/lib"

echo "=== Building Lua 5.3.5 ==="
if [ ! -f "$LUA_DIR/src/liblua.a" ]; then
    cd "$LUA_DIR/src"
    # Build only the static library (not lua/luac executables which need readline)
    make liblua.a CC=g++ CFLAGS='-O2 -fPIC -DLUA_USE_LINUX' SYSCFLAGS='-DLUA_USE_LINUX' SYSLIBS='-ldl'
    echo "Lua built successfully."
else
    echo "Lua already built, skipping."
fi

echo ""
echo "=== Building ygopro-core ==="
mkdir -p "$OUT_DIR"
cd "$CORE_DIR"

g++ -shared -fPIC -o "$OUT_DIR/libocgcore.so" \
    *.cpp \
    -I"$LUA_DIR/src" \
    -L"$LUA_DIR/src" \
    -llua \
    -std=c++14 \
    -O2

echo "libocgcore.so built successfully at $OUT_DIR/libocgcore.so"
echo ""

# Verify
echo "=== Verifying ==="
ls -la "$OUT_DIR/libocgcore.so"
echo "Done!"
