# ClusterFuzzLite builder image for Vaara fuzz targets.
#
# Base image is OSS-Fuzz's `base-builder-python`, which already carries
# atheris + compile_python_fuzzer plumbing. We layer Vaara's source tree
# on top and install with the [yaml,attestation] extras so the policy
# loader and OVERT envelope targets exercise the real PyYAML / cbor2
# / cryptography code paths.
FROM gcr.io/oss-fuzz-base/base-builder-python

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY . $SRC/vaara
WORKDIR $SRC/vaara

COPY .clusterfuzzlite/build.sh $SRC/build.sh
