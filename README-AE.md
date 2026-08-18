# PyTorch-Chipyard Artifact Evaluation

This README contains the single end-to-end command sequence used for artifact
evaluation on the authors' preconfigured FPGA server.

## One-time server prerequisite

Before the AE account is handed to a reviewer, the server administrator must
have already configured:

- Chipyard/FireSim, the FPGA bitstreams, XRT/Vivado, and FPGA device access;
- the RISC-V toolchain and the TVM-Gemmini/Verilator environment used by
  Table 2;
- rootless Docker access for the AE account;
- passwordless SSH from the AE account to `localhost` for the local run farm;
- shared FireMarshal/FireSim output directories writable by the `firesim`
  group; and
- `CHIPYARD_DIR` and `TABLE2_TVM_AE_ROOT` defined by the AE account's
  `~/.ae-env.sh`, with that file loaded by `.bashrc`.

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

Stage 1 compiles the paper workloads and records PyTorch-Chipyard's Docker-side
compile measurements for three sampled Table 2 kernels derived from SqueezeNet,
ResNet-50, and AlexNet. Their fixed definitions are maintained by
`scripts/table2_results.py`.

## 3. Run Stage 2

If Stage 2 is started in a new shell, load the preconfigured account
environment once before running it. `run-stage2.sh` also sources
`~/.ae-env.sh` automatically, so the host paths are identical in every shell.

```bash
source ~/.bashrc
cd ~/pytorch-chipyard

bash scripts/run-stage2.sh
```

Stage 2 builds the ELF files and FireMarshal images, runs the ordinary FireSim
workloads, and completes Table 2 with PyTorch-Chipyard/FireSim and
TVM-Gemmini/Verilator. It builds the Verilator simulator once if necessary;
that setup cost is not included in the table. RVV guest kernel failures are
retried automatically; other failures stop the workflow.

To reproduce only the bounded Table 2 experiment, use the same Docker mount
options as above and replace the Stage 1 command with:

```bash
bash scripts/run-stage1.sh --only-table2
```

Then run the host part with:

```bash
bash scripts/run-stage2.sh --only-table2
```

The experiment intentionally uses each tool's native target:
PyTorch-Chipyard uses FP32, a DIM=8 Gemmini, four Rocket cores, and FireSim;
TVM-Gemmini uses INT8, a DIM=16 Gemmini, one Rocket core, and Verilator.
Consequently, the reported values characterize practical compile and
cycle-accurate RTL turnaround for each workflow; they must not be interpreted
as a cross-tool kernel-performance speedup. The TVM path compiles against the
Gemmini headers generated for the selected Chipyard Verilator target and
rejects a run if those headers have drifted, so the older headers vendored by
TVM-Gemmini cannot silently mismatch the RTL simulator.

## 4. Generate the paper outputs

```bash
source ~/.bashrc
cd ~/pytorch-chipyard

bash scripts/run-plot.sh
```

Generated figures are written as `scripts/figures/fig*.pdf`. A complete Table 2
run writes `scripts/figures/table2.csv` and `scripts/figures/table2_rows.tex`.
Raw logs, target metadata, compiler artifacts, and intermediate results remain
under `results/table2/`.
