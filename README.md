# pytorch-chipyard
A reusable compiler-stack foundation for ML research on Chipyard hardware.

This README focuses on installation commands and the artifact replication flow
for the paper evaluation. The documentation under `docs/` is not fully updated
for the current Docker and external-Chipyard workflow yet.

https://pytorch-chipyard.readthedocs.io/en/latest/

## Installation

The conda version below is the version used by the authors. Other conda
versions may work, but `24.11.3` is the validated baseline. This repository only
contains the pytorch-chipyard compiler stack used to compile PyTorch models into
artifacts that can later run through Chipyard. End-to-end FPGA execution still
requires a separate Chipyard installation, FireSim local FPGA setup, Vivado/XRT,
and the matching FPGA bitstreams. Prebuilt bitstreams are distributed through
the GitHub release artifacts:

https://github.com/JongseoKang/pytorch-chipyard/releases

```bash
# pre-requisite: conda-24.11.3
git clone https://github.com/JongseoKang/pytorch-chipyard

cd pytorch-chipyard
git submodule update --init pytorch triton triton_chipyard llvm-project buddy-mlir

bash scripts/install.sh
```

### Docker Installation

The Docker setup is an alternative Stage 1 compiler installation path. It does
not install Chipyard, FireMarshal, FireSim, Vivado, XRT, or FPGA host drivers.

```bash
git clone https://github.com/JongseoKang/pytorch-chipyard

cd pytorch-chipyard
git submodule update --init pytorch triton triton_chipyard llvm-project buddy-mlir

# This may require sudo depending on the host Docker setup. If Docker is run
# with sudo, bind-mounted artifacts may need ownership repair before Stage 2.
docker build -f docker/stage1.Dockerfile -t pytorch-chipyard:stage1 .
```

The image uses Ubuntu 20.04 and installs the Python 3.12 Miniconda bootstrap
under `/root/anaconda3`, then pins base conda to `24.11.3` before running
`scripts/install.sh`. The installed runtime defaults to the `pytorch-chipyard`
conda environment. If Docker daemon access fails with a permission error, run
the build through the host's approved Docker access path, for example `sudo
docker build ...` or a configured `docker` group.

### Local FPGA Host Prerequisite

The installation commands above initialize this repository's compiler stack.
They do not install Chipyard/FireSim or configure the Linux host as a FireSim
FPGA run farm machine. Local FPGA support is a separate machine-level
prerequisite because it installs privileged helper scripts under
`/usr/local/bin` and configures device access for the FPGA host.

On the authors' artifact review server, this prerequisite is already handled.
The checked Chipyard repository is based on commit
`dbc082e2206f787c3aba12b9b171e1704e15b707` with local hardware/configuration
changes. The evaluated FireSim checkout uses commit
`141bff735c9b81a6d82a593310d0ca8be903e9a1`, which is on FireSim `main` after
the `1.20.1` release. Use the FireSim `latest` local FPGA setup document for
the rendered guide:
[FireSim Local FPGA Initial Setup](https://docs.fires.im/en/latest/Local-FPGA-Initial-Setup.html).
For an exact source permalink matching this checkout, see
[docs/Local-FPGA-Initial-Setup.rst at 141bff735](https://github.com/firesim/firesim/blob/141bff735c9b81a6d82a593310d0ca8be903e9a1/docs/Local-FPGA-Initial-Setup.rst).

pytorch-chipyard compiler artifacts are not tightly coupled to that exact
Chipyard commit. In practice, the pytorch-chipyard compile flow requires
`riscv64-unknown-linux-gnu-g++`; the version used by the authors is 12.2.0
(g2ee5e430018), and this version is recommended. If toolchain-related errors
occur, prefer adding a separate RISC-V toolchain rather than changing the
Chipyard version selected for the local FPGA setup. Use a Chipyard/FireSim
version that matches your FPGA board, Vivado/XRT installation, and bitstream
setup.

If your Chipyard checkout's `env.sh` selects a different RISC-V GCC, keep
`CHIPYARD_DIR` pointed at that Chipyard checkout and set
`PYTORCH_CHIPYARD_RISCV_TOOLCHAIN_DIR` to a separate toolchain root or `bin/`
directory containing `riscv64-unknown-linux-gnu-g++`.

## Artifact Replication

Artifact replication has two stages. Stage 1 compiles PyTorch models and
generates portable compiler artifacts. Stage 2 consumes those artifacts,
builds `model.elf`, packages FireMarshal workloads, and runs them through
FireSim.

### Stage 1: Torch Model Compile

```bash
# Source-build version.
mkdir -p examples
bash scripts/run-stage1.sh

# Docker-build version.
mkdir -p examples
docker run --rm -it \
  -v "$PWD/examples:/opt/pytorch-chipyard/examples" \
  -v pytorch-chipyard-triton-cache:/tmp/triton-chipyard-cache \
  pytorch-chipyard:stage1 \
  bash scripts/run-stage1.sh
```

After either path, the paper workload artifacts are written under
`pytorch-chipyard/examples/artifact-<model>/<backend>/` or a deeper
experiment-specific path such as
`examples/artifact-opt/gemmini/window/seq1024/`. The
gemmini-max-autotune workload uses
`examples/artifact-gemmini-max-autotune/gemmini/`.

### Stage 2: FireMarshal Workload Packaging and FireSim FPGA Simulation

Stage 2 uses the Stage 1 artifacts, such as `runner.cpp`, `weights.bin`,
`input.bin`, and staged kernel objects, to build `model.elf` files. It then
generates FireMarshal workloads and runs them through FireSim. This stage
requires the local FPGA host setup described above.

```bash
# Author review server setting.
export CHIPYARD_DIR=/home/hongjun/hk_chipyard/chipyard
export PYTORCH_CHIPYARD_FPGA_DB=/opt/firesim-db0.json
# Optional: set only when model.elf builds should use a separate RISC-V GCC.
# export PYTORCH_CHIPYARD_RISCV_TOOLCHAIN_DIR=/path/to/riscv-gcc-12.2.0

# For other hosts, replace the author review Chipyard path above:
# export CHIPYARD_DIR=/path/to/chipyard
# For other hosts, replace the author review FPGA DB path above:
# export PYTORCH_CHIPYARD_FPGA_DB=/path/to/firesim-db.json
# See scripts/env.sh for advanced Chipyard/FireMarshal/FireSim/toolchain overrides.

source scripts/env.sh

# If privileged Docker generated root-owned artifacts, fix host ownership
# sudo chown -R "$USER:$USER" examples

bash scripts/run-stage2.sh

# Generate paper figures from collected FireSim results.
bash scripts/run-plot.sh
```

## Citation
