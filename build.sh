#!/usr/bin/env bash
# IECNN Build Script — compile all C modules into shared libraries
# Libraries use _c.so suffix to avoid conflicts with Python module names
# Usage: bash build.sh

set -e
GCC=$(which gcc 2>/dev/null || echo "/nix/store/s41bqqrym7dlk8m3nk74fx26kgrx0kv8-replit-runtime-path/bin/gcc")
FLAGS="-O2 -shared -fPIC -lm"

echo "[build] Compiling formulas.c → formulas/formulas_c.so"
$GCC $FLAGS -o formulas/formulas_c.so formulas/formulas.c

echo "[build] Compiling basemapping.c → basemapping/basemapping_c.so"
$GCC $FLAGS -o basemapping/basemapping_c.so basemapping/basemapping.c

echo "[build] Compiling aim.c → aim/aim_c.so"
$GCC $FLAGS -o aim/aim_c.so aim/aim.c

echo "[build] Compiling convergence.c → convergence/convergence_c.so"
$GCC $FLAGS -I. -o convergence/convergence_c.so convergence/convergence.c formulas/formulas.c

echo "[build] All C extensions built successfully."
