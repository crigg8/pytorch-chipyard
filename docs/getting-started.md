# Section 1: Installation

Follow the installation flow in the repository README. The documented setup
assumes the versions pinned by this repository.

```bash
# pre-requisite: conda-24.11.3
git clone https://github.com/crigg8/pytorch-chipyard.git

# source build
cd pytorch-chipyard
git submodule update --init pytorch triton triton_chipyard llvm-project buddy-mlir chipyard

bash scripts/install.sh

# docker build
# This may require sudo depending on the host Docker setup. 
# If Docker is run with sudo, bind-mounted artifacts may need ownership.
docker build -f docker/stage1.Dockerfile -t pytorch-chipyard:stage1 .
```

`scripts/install.sh` builds the compiler stack provided by the
pytorch-chipyard repository. This stack lowers PyTorch models into RISC-V
kernels that can later be packaged for Chipyard-based execution.

- Creates a `pytorch-chipyard` conda environment with Python 3.12.
- Installs PyTorch 2.8.0, torchvision 0.23.0, Triton build dependencies, and
  Buddy-MLIR dependencies.
- Builds the pinned `llvm-project` checkout with MLIR, LLVM, LLD, and RISC-V
  target support.
- Installs the pinned Triton checkout with `TRITON_PLUGIN_DIRS` pointing to the
  out-of-tree `triton_chipyard` backend.
- Builds the pinned `buddy-mlir` checkout against the local LLVM/MLIR build.
- Replaces the installed PyTorch package's `torch/_inductor` with the custom
  `pytorch/torch/_inductor` from this repository.

## Repository Scope

This repository includes only the compiler stack. It does not install chipyard or configure an FPGA host.
For FPGA execution, the host must already provide the FPGA runtime environment.

External FPGA-side requirements include:

- An AMD/Xilinx Alveo U250-class host for the full paper configuration. Some
  experiments may also run on an Alveo U280 if matching bitstreams are
  available.
- XRT and the XDMA kernel module.
- Vivado/Vitis-compatible tooling for the target bitstream flow.
- FireSim database files and access to the target FPGA device.
- Prebuilt bitstreams from the release artifacts, or bitstreams built from the
  matching Chipyard/FireSim hardware configuration.

FPGA setup and FPGA execution are outside the coverage of this documentation.
The examples here stop at compiler-stack setup, artifact generation, and the
commands needed to build generated RISC-V runner binaries.

## Version Assumptions

This project is version-sensitive. The documented path assumes:

- The custom PyTorch checkout included by this repository.
- The custom LLVM checkout included by this repository.
- The custom Buddy-MLIR checkout included by this repository.
- The custom Triton-Chipyard backend checkout included by this repository.
- The pinned Triton submodule commits.

Other commits may work, but they are outside the documented configuration.
