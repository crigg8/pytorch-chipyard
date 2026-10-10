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

## Prerequisites

This repository includes only the compiler stack. It does not install Chipyard or configure an FPGA host.
For FPGA execution, the host must already provide the FPGA runtime environment.
Follow the guides below to set up FireSim simulation. 

- **Chipyard:** [Initial repository setup][chipyard-setup]
- **FireSim:** [FPGA host setup][firesim-host] ·
  [Repository and manager setup][firesim-manager] ·
  [U250 Vivado configuration][firesim-vivado]
- **FireMarshal:** [Installation][firemarshal-setup] ·
  [Quick start][firemarshal-quickstart] ·
  [Configuration reference][firemarshal-config]

## Version Assumptions

This project is version-sensitive. The documented path assumes:

- The custom PyTorch checkout included by this repository.
- The custom LLVM checkout included by this repository.
- The custom Buddy-MLIR checkout included by this repository.
- The custom Triton-Chipyard backend checkout included by this repository.
- The pinned Triton submodule commits.

Other commits may work, but they are outside the documented configuration.

[chipyard-setup]: https://github.com/ucb-bar/chipyard/blob/dbc082e2206f787c3aba12b9b171e1704e15b707/docs/Chipyard-Basics/Initial-Repo-Setup.rst
[firesim-host]: https://github.com/firesim/firesim/blob/141bff735c9b81a6d82a593310d0ca8be903e9a1/docs/Local-FPGA-Initial-Setup.rst
[firesim-manager]: https://github.com/firesim/firesim/blob/141bff735c9b81a6d82a593310d0ca8be903e9a1/docs/Getting-Started-Guides/On-Premises-FPGA-Getting-Started/Repo-Setup/Xilinx-XDMA-Template.rst
[firesim-vivado]: https://github.com/firesim/firesim/blob/141bff735c9b81a6d82a593310d0ca8be903e9a1/docs/Getting-Started-Guides/On-Premises-FPGA-Getting-Started/Building-a-FireSim-Bitstream/Xilinx-Alveo-U250.rst
[firemarshal-setup]: https://github.com/firesim/FireMarshal/blob/21119e5ce922ff9302f0bc9d2d7349c77f7ac064/README.md#setup
[firemarshal-quickstart]: https://github.com/firesim/FireMarshal/blob/21119e5ce922ff9302f0bc9d2d7349c77f7ac064/docs/source/Tutorials/quickstart.rst
[firemarshal-config]: https://github.com/firesim/FireMarshal/blob/21119e5ce922ff9302f0bc9d2d7349c77f7ac064/docs/source/marshalConfig.rst
