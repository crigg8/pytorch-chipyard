# PyTorch-Chipyard

A reusable compiler-stack foundation for ML research on Chipyard hardware.

## Documentation

The full documentation is available at
[pytorch-chipyard.readthedocs.io](https://pytorch-chipyard.readthedocs.io/en/latest/).
For artifact evaluation and paper reproduction, see
[README-AE.md](README-AE.md).

## Installation

The validated source installation uses Conda 24.11.3 with Python 3.12. The
installation script expects Conda under `~/anaconda3`.

```bash
git clone https://github.com/crigg8/pytorch-chipyard.git
cd pytorch-chipyard

git submodule update --init \
  pytorch triton triton_chipyard llvm-project buddy-mlir

bash scripts/install.sh

# Activate the installed environment in a new shell.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch-chipyard
```

This installs the PyTorch-Chipyard compiler stack. FPGA execution additionally
requires a compatible Chipyard/FireSim checkout, Vivado/XRT, FPGA drivers, and
matching bitstreams; those machine-level components are not installed by this
script.

### Docker Installation

Docker provides a self-contained alternative for the compiler installation.
It does not install or configure the FPGA host components listed above.

```bash
git clone https://github.com/crigg8/pytorch-chipyard.git
cd pytorch-chipyard

git submodule update --init \
  pytorch triton triton_chipyard llvm-project buddy-mlir

docker build \
  -f docker/stage1.Dockerfile \
  -t pytorch-chipyard:stage1 \
  .

# Open a shell in the installed environment.
docker run --rm -it pytorch-chipyard:stage1
```

Depending on the host Docker configuration, the `docker` commands may require
`sudo` or a configured rootless Docker daemon.
