#!/usr/bin/env bash
set -e
GCC=$(which gcc 2>/dev/null || echo "gcc")
FLAGS="-O3 -shared -fPIC -lm -march=native"

echo "[build] Compiling formulas.c → formulas/formulas_c.so"
$GCC $FLAGS -o formulas/formulas_c.so formulas/formulas.c

echo "[build] Compiling basemapping.c → basemapping/basemapping_c.so"
$GCC $FLAGS -o basemapping/basemapping_c.so basemapping/basemapping.c

echo "[build] Compiling aim.c → aim/aim_c.so"
$GCC $FLAGS -o aim/aim_c.so aim/aim.c

echo "[build] Compiling convergence.c → convergence/convergence_c.so"
$GCC $FLAGS -I. -o convergence/convergence_c.so convergence/convergence.c formulas/formulas.c

echo "[build] Compiling pruning.c → pruning/pruning_c.so"
$GCC $FLAGS -I. -o pruning/pruning_c.so pruning/pruning.c formulas/formulas.c

echo "[build] Compiling neural_dot.c → neural_dot/neural_dot_c.so"
$GCC $FLAGS -I. -o neural_dot/neural_dot_c.so neural_dot/neural_dot.c formulas/formulas.c

echo "[build] Compiling decoder.c → decoding/decoder_c.so"
$GCC $FLAGS -o decoding/decoder_c.so decoding/decoder.c

echo "[build] All C extensions built successfully."
