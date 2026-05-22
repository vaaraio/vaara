#!/bin/bash -eu
# ClusterFuzzLite build script for Vaara fuzz targets.
#
# Installs Vaara with the optional extras the fuzz targets actually touch
# (attestation = cbor2 + cryptography; yaml = pyyaml), then compiles each
# `fuzz/fuzz_*.py` target with `compile_python_fuzzer` from base-builder.

pip3 install --no-cache-dir ".[attestation,yaml]"

for fuzzer in "$SRC/vaara/fuzz/"fuzz_*.py; do
    compile_python_fuzzer "$fuzzer"
done
