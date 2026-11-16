# BakeLab PyTorch wheel release

`.github/workflows/gitea-cuda-wheel.yml` builds and publishes the PyTorch
CUDA wheel when `trim` advances in BakeLab Gitea. It runs on the
`[self-hosted, linux, x64, bb, cuda]` runner and uses PyTorch's own
`.ci/pytorch/binary_populate_env.sh` and `.ci/wheel/linux/build.sh` manywheel
pipeline inside the digest-pinned CUDA 13.4 builder image.

The workflow currently publishes one Python 3.14 CUDA 13.4 wheel to the Gitea
PyPI registry at `https://gitea.int.bb.pub/api/packages/BakeLab/pypi/`. It
requests a short-lived Actions OIDC identity for the Gitea Packages audience
and supplies it to `uv publish`; no long-lived upload secret is used. The
Gitea server must grant this repository's workflow permission to publish to
the `BakeLab/pypi` package index.

PyTorch's build script encodes the UTC build date, CUDA variant, and source
commit prefix in the wheel version. The job also retains full source and
submodule SHAs, builder image digest, compiler and CUDA compiler versions,
wheel hashes, and run ID as a provenance artifact. Native AOT embedding is
disabled for this base wheel, so the job does not build or publish Triton or
other separately maintained components.

The separate `cuda-ci.yml` job remains the CUDA source-build and GPU smoke
test. It is not a release workflow. The inherited upstream S3/nightly binary
release workflows and hardware paths with no BakeLab runner are removed; the
portable source checks and BakeLab CUDA CI remain.
