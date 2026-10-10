# PyTorch-Chipyard Artifact Evaluation

This README contains the single end-to-end command sequence used for artifact
evaluation on the authors' preconfigured FPGA server.

For dependency versions and setup references, see
[Prerequisites and Installation](https://pytorch-chipyard.readthedocs.io/en/latest/getting-started.html).
On the provided setup, the smoke tests take approximately **five hours**, and
the simplified Stage 2 evaluation takes approximately **three days**. These
are estimates for an already configured environment; installation, Stage 1
compilation for the paper workloads, and host load affect the overall time.

## One-time server prerequisite

Before the AE account is handed to a reviewer, the server administrator must
have already configured:

- Chipyard/FireSim, the FPGA bitstreams, XRT/Vivado, and FPGA device access;
- the RISC-V toolchain and the TVM-Gemmini/Verilator environment used by
  Table 4.

Once this environment is configured, reviewers can reuse it for all workloads
by loading the account environment in each new shell. The artifact hardware
revisions are Chipyard `dbc082e2`, FireSim `141bff735`, and FireMarshal `21119e5`.

## 1. Clone and build the Stage 1 image

```bash
git clone https://github.com/crigg8/pytorch-chipyard.git
cd pytorch-chipyard

git submodule update --init \
  pytorch triton triton_chipyard llvm-project buddy-mlir

source ~/.bashrc

docker build \
  -f docker/stage1.Dockerfile \
  -t pytorch-chipyard:stage1 \
  .
```

The Docker image only builds the compiler environment used by Stage 1. The
external FPGA host environment is supplied by the preconfigured server.
A Dockerized version of the complete workflow is being validated, including
long-running hardware evaluations. It is planned for the camera-ready version,
together with setup and usage instructions, and is not yet available.

## 2. Run the bounded smoke test

Before starting the full evaluation, compile the shared 32x32x32 GEMM for the
three PyTorch-Chipyard targets inside the Stage 1 image:

```bash
mkdir -p results

docker run --rm -it \
  -v "$PWD/results:/opt/pytorch-chipyard/results" \
  -v pytorch-chipyard-triton-cache:/tmp/triton-chipyard-cache \
  pytorch-chipyard:stage1 \
  bash scripts/run-smoke-test-stage1.sh
```

The command compiles separate RVV, Gemmini, and scalar artifacts and prints
`SMOKE_STAGE1_STATUS=PASS`. Complete the smoke test on the preconfigured FPGA
host with:

```bash
source ~/.bashrc
cd ~/pytorch-chipyard

bash scripts/run-smoke-test.sh
```

This runs the GEMM through TVM-Gemmini on its single-core INT8 DIM=16
Verilator target, and through PyTorch-Chipyard on the RVV four-core, Gemmini
four-core, and scalar Rocket sixteen-core FireSim targets. The test checks
successful simulator completion and the expected runtime artifacts rather
than comparing numerical results. It prints the path to every successful run
and ends with `SMOKE_TEST_STATUS=PASS`. Results and full logs are written under
`results/smoke-test/`, with the latest successful run linked as
`results/smoke-test/latest`.

The Verilator simulator is part of the preconfigured host setup. If it needs
to be rebuilt, use `bash scripts/run-smoke-test.sh --build-verilator`.

## 3. Run Stage 1

```bash
mkdir -p examples results

docker run --rm -it \
  -v "$PWD/examples:/opt/pytorch-chipyard/examples" \
  -v "$PWD/results:/opt/pytorch-chipyard/results" \
  -v pytorch-chipyard-triton-cache:/tmp/triton-chipyard-cache \
  pytorch-chipyard:stage1 \
  bash scripts/run-stage1.sh
```

Stage 1 compiles the paper workloads and records PyTorch-Chipyard's Docker-side
compile measurements for three sampled Table 4 kernels derived from SqueezeNet,
ResNet-50, and MobileNetV2. Their fixed definitions are maintained by
`scripts/table4_results.py`.

A successful run prints `STAGE1_STATUS=PASS` together with the artifact and
result directories.

## 4. Run Stage 2

If Stage 2 is started in a new shell, load the preconfigured account
environment once before running it. `run-stage2.sh` also sources
`~/.ae-env.sh` automatically, so the host paths are identical in every shell.

After Stage 1, use the simplified Stage 2 workflow for a smaller selection of
experiments. It takes approximately three days on the provided setup:

```bash
# Simple test: Partial Figures 6ab, 8a, 9, 11, 13ac
bash scripts/simple-stage2.sh
```

This selection runs SqueezeNet and OPT-125M, the Figure 11 blocking case
`(64, 64, 512)`, and OPT-125M with sequence length 256. It generates partial
Figures 6(a,b), 8(a), 9, 11, and 13(a,c), and finishes with
`SIMPLE_STAGE2_STATUS=PASS`. It checks a subset of the artifact workflow;
reproducing all paper results requires the corresponding full experiment units.
Use `--skip-plot` to run the selected workloads without plotting.

For the full results, choose any of the five independently resumable experiment
units below. Some figures share workloads and are grouped into one unit:

```bash
# Figures 7, 8, and 9, plus Table 5
bash scripts/run-stage2.sh --experiment=figures-7-8-9-table5

# Figure 10
bash scripts/run-stage2.sh --experiment=figure-10

# Figure 11
bash scripts/run-stage2.sh --experiment=figure-11

# Figure 13
bash scripts/run-stage2.sh --experiment=figure-13

# Table 4
bash scripts/run-stage2.sh --experiment=table-4
```

The units may be run in any order and may be rerun after an interruption. For
ordinary FireSim workloads, Stage 2 checks
`scripts/figures/results-workload/<workload>/` before doing work. A workload is
skipped when its completion marker, `model.log`, `autotune.log`, and required
output artifact are already complete. This also avoids rerunning shared
experiments: for example, Figure 10 reuses the direct-convolution Gemmini runs
from Figures 7--9, and Figure 13 reuses the 256-token SDPA runs from Table 5.
Table 4 similarly resumes completed measurements recorded in
`results/table4/stage1-latest/raw.csv`.

The simplified and full Stage 2 flows reuse completed shared workloads. The
reduced Figure 11 run has a separate completion record, so it never marks the
full Figure 11 experiment complete. The simplified flow builds, runs, and
removes one FireMarshal image at a time to reduce disk usage.

The original one-shot command remains available when a single uninterrupted
run is preferred:

```bash
bash scripts/run-stage2.sh
```

Stage 2 builds the required ELF files and FireMarshal images, runs the selected
FireSim workloads, and completes Table 4 with PyTorch-Chipyard/FireSim and
TVM-Gemmini/Verilator.
Stage 2 validates workflow completion rather than comparing model output with
an eager-mode numerical reference. Each FireSim workload must terminate
successfully and produce its expected `model.log`, `autotune.log`, and output
artifact.
A successful run prints their paths and ends with `STAGE2_STATUS=PASS`. 

## 5. Generate the paper outputs

```bash
source ~/.bashrc
cd ~/pytorch-chipyard

bash scripts/run-plot.sh
```

Generated figures use the semantic filenames referenced by the paper source
under `scripts/figures/`. The plotting workflow checks every generated plot
referenced by the paper, prints each path, and ends with `FIGURES_STATUS=PASS`.
A complete Table 4 run writes `scripts/figures/table4.csv` and
`scripts/figures/table4_rows.tex`. Raw logs, target metadata, compiler artifacts,
and intermediate results remain under `results/table4/`.
