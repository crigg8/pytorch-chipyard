# PyTorch-Chipyard Artifact Evaluation

This README contains the single end-to-end command sequence used for artifact
evaluation on the authors' preconfigured FPGA server.

## One-time server prerequisite

Before the AE account is handed to a reviewer, the server administrator must
have already configured:

- Chipyard/FireSim, the FPGA bitstreams, XRT/Vivado, and FPGA device access;
- the RISC-V toolchain and the TVM-Gemmini environment used by Table 2;
- rootless Docker access for the AE account;
- passwordless SSH from the AE account to `localhost` for the local run farm;
- shared FireMarshal/FireSim output directories writable by the `firesim`
  group; and
- the environment loaded by the AE account's `.bashrc`.

These are host/account setup tasks performed once by the administrator. They
are not repeated for each clone or experiment. On the authors' review server,
they are already complete.

## 1. Clone and build the Stage 1 image

```bash
git clone --recurse-submodules https://github.com/crigg8/pytorch-chipyard.git
cd pytorch-chipyard
source ~/.bashrc

docker build \
  -f docker/stage1.Dockerfile \
  -t pytorch-chipyard:stage1 \
  .
```

The Docker image only builds the compiler environment used by Stage 1. The
external FPGA host environment is supplied by the preconfigured server.

## 2. Run Stage 1

```bash
mkdir -p examples results

docker run --rm -it \
  -v "$PWD/examples:/opt/pytorch-chipyard/examples" \
  -v "$PWD/results:/opt/pytorch-chipyard/results" \
  -v pytorch-chipyard-triton-cache:/tmp/triton-chipyard-cache \
  pytorch-chipyard:stage1 \
  bash scripts/run-stage1.sh
```

Stage 1 compiles the paper workloads and records the Docker-side Table 2
compile measurements.

## 3. Run Stage 2

If Stage 2 is started in a new shell, load the preconfigured account
environment once before running it.

```bash
source ~/.bashrc
cd ~/pytorch-chipyard

bash scripts/run-stage2.sh
```

Stage 2 builds the ELF files and FireMarshal images, runs the FireSim workloads,
and completes Table 2. RVV guest kernel failures are retried automatically;
other failures stop the workflow.

## 4. Generate the paper outputs

```bash
source ~/.bashrc
cd ~/pytorch-chipyard

bash scripts/run-plot.sh
```

Generated figures are written as `scripts/figures/fig*.pdf`. Table 2 is written
as `scripts/figures/table2.csv`. Raw logs, compiler artifacts, and intermediate
results remain under `examples/` and `results/`.
