# PyTorch-Chipyard Artifact Evaluation
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

https://github.com/crigg8/pytorch-chipyard/releases

```bash
# pre-requisite: conda-24.11.3
git clone https://github.com/crigg8/pytorch-chipyard.git

cd pytorch-chipyard
git submodule update --init pytorch triton triton_chipyard llvm-project buddy-mlir

bash scripts/install.sh
```

### Docker Installation

The Docker setup is an alternative Stage 1 compiler installation path. It does
not install Chipyard, FireMarshal, FireSim, Vivado, XRT, or FPGA host drivers.

```bash
git clone https://github.com/crigg8/pytorch-chipyard.git

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
mkdir -p examples results
docker run --rm -it \
  -v "$PWD/examples:/opt/pytorch-chipyard/examples" \
  -v "$PWD/results:/opt/pytorch-chipyard/results" \
  -v pytorch-chipyard-triton-cache:/tmp/triton-chipyard-cache \
  -v pytorch-chipyard-torch-cache:/root/.cache/torch \
  pytorch-chipyard:stage1 \
  bash scripts/run-stage1.sh
```

Stage 1 runs inside the Docker image built from the current checkout. The paper
workload artifacts are written under
`pytorch-chipyard/examples/artifact-<model>/<backend>/` or a deeper
experiment-specific path such as
`examples/artifact-opt/gemmini/window/seq1024/`. The
gemmini-max-autotune workload uses
`examples/artifact-gemmini-max-autotune/gemmini/`.

The default Docker Stage 1 flow also measures the fresh PyTorch-Chipyard CNN
compilations used by Table 2. Its raw timing rows and compiler artifacts are
stored under `results/table2/<UTC-run-id>-stage1/`; the portable
`results/table2/stage1-latest` symlink identifies the run that Stage 2 must
complete. To test only this path, optionally with one model, use:

```bash
docker run --rm -it \
  -v "$PWD/results:/opt/pytorch-chipyard/results" \
  -v pytorch-chipyard-triton-cache:/tmp/triton-chipyard-cache \
  -v pytorch-chipyard-torch-cache:/root/.cache/torch \
  pytorch-chipyard:stage1 \
  bash scripts/run-stage1.sh --only-table2 --table2-models=alexnet
```

The default Stage 1 flow also generates the alias-first ablation. For AlexNet,
MobileNetV2, ResNet50, and SqueezeNet it generates the alias-first OFF artifact;
the ordinary Gemmini artifact is the ON case. For GPT-2, GPT-Neo, OPT, and
Pythia it generates both ON and OFF with sequence length 256 and SDPA. All
ablation workloads use fp32 8x8 Gemmini and 4 cores. Use `--only-alias-first`
to compile just this experiment or `--skip-alias-first` to omit it from the
complete flow.

### Stage 2: FireMarshal Workload Packaging and FireSim FPGA Simulation

Stage 2 uses the Stage 1 artifacts, such as `runner.cpp`, `weights.bin`,
`input.bin`, and staged kernel objects, to build `model.elf` files. It then
generates FireMarshal workloads and runs them through FireSim. For Table 2 it
reuses the fresh PyTorch-Chipyard compilation measured inside Docker Stage 1,
measures the TVM-Gemmini baseline in its own prepared environment, and appends
the Spike and FireSim host-wall measurements to the same `raw.csv`. This stage
requires the local FPGA host setup described above.

```bash
# Author review server setting.
export CHIPYARD_DIR=/home/hongjun/hk_chipyard/chipyard
export PYTORCH_CHIPYARD_FPGA_DB=/opt/firesim-db0.json
export PYTORCH_CHIPYARD_RISCV_TOOLCHAIN_DIR=/home/hongjun/miniforge3/pkgs/riscv-tools-1.0.3-0_h1234567_ga1b1b14/riscv-tools
export TABLE2_TVM_AE_ROOT=/home/ae/tvm-gemmini-ae

# For other hosts, replace the author review Chipyard path above:
# export CHIPYARD_DIR=/path/to/chipyard
# For other hosts, replace the author review FPGA DB path above:
# export PYTORCH_CHIPYARD_FPGA_DB=/path/to/firesim-db.json
# For other hosts, set this only when model.elf builds should use a separate
# RISC-V GCC 12.2.0 toolchain:
# export PYTORCH_CHIPYARD_RISCV_TOOLCHAIN_DIR=/path/to/riscv-gcc-12.2.0
# See scripts/env.sh for advanced Chipyard/FireMarshal/FireSim/toolchain overrides.

source scripts/env.sh

# If privileged Docker generated root-owned artifacts, fix host ownership
# sudo chown -R "$USER:$USER" examples results

bash scripts/run-stage2.sh

# Generate paper figures from collected FireSim results.
bash scripts/run-plot.sh
```

The default `run-stage2.sh` command completes the Table 2 run started by Docker
Stage 1 after the ordinary Stage 2 workloads finish. Its paper-facing result is
`scripts/figures/table2.csv`; the same file is archived as
`results/table2/<UTC-run-id>-stage1/table2.csv`. `raw.csv`, command logs, and the
per-trial artifacts are stored beside the archived file. Table 2 is a table
rather than a PDF figure, so `run-plot.sh` does not transform this CSV.

Table 2 compares this repository with a separately prepared TVM-Gemmini AE
tree. The author review server already provides that tree at the path above.
On another host, prepare the equivalent TVM-Gemmini environment and point
`TABLE2_TVM_AE_ROOT` at its root. The tree must contain `scripts/env.sh`, the
AE-local model ports, TVM build, RISC-V tools, and Gemmini-compatible Spike
described by that artifact. Stage 2 checks this prerequisite before starting a
full run, rather than failing after the FPGA experiments have completed. Use
`--skip-table2` only when intentionally reproducing the other paper results
without Table 2.

Selective Stage 2 commands (`--workload`, `--only-alias-first`, or
`--only-alias-first-cnn-off`) and `--skip-firesim` do not launch the unrelated
full Table 2 matrix. To reproduce only Table 2, first run Docker Stage 1 with
`--only-table2`, then run the host command below. Use the same
`--table2-models` and `--table2-repeats` values in both commands when selecting
a smaller smoke test.

```bash
bash scripts/run-stage2.sh --only-table2 --table2-models=alexnet
```

The completed default matrix covers all four CNN models, both toolchains, and
the compile, Spike, and FireSim phases. `run_table2.sh --resume` is the lower-
level continuation interface used by Stage 2; it is not necessary in the
normal AE command sequence.

`run-plot.sh` names generated plots by their location in the paper:

| Paper panel | Output |
| --- | --- |
| Figure 6(a), 6(b), 6(c) | `fig6a.pdf`, `fig6b.pdf`, `fig6c.pdf` |
| Figure 7(a), 7(b), 7(c) | `fig7a.pdf`, `fig7b.pdf`, `fig7c.pdf` |
| Figure 8(a), 8(b) | `fig8a.pdf`, `fig8b.pdf` |
| Figure 9(a), 9(b), 9(c) | `fig9a.pdf`, `fig9b.pdf`, `fig9c.pdf` |
| Figure 10 | `fig10.pdf` |
| Figure 12(a), 12(b), 12(c) | `fig12a.pdf`, `fig12b.pdf`, `fig12c.pdf` |

All of these files are written under `scripts/figures/`. Figure 11 is the
FlexAttention pseudocode typeset directly in the paper and therefore has no
plot-script output.

RVV workloads are monitored while FireSim is running. If UART reports a kernel
panic, kernel Oops, or the known repeated-byte corruption, Stage 2 runs
`firesim kill`, preserves the last 4 MiB of the failed UART under
`$PYTORCH_CHIPYARD_LOG_DIR/failed-attempts/` (by default
`examples/.logs/failed-attempts/`), clears `FIRESIM_RUNS_DIR`, and retries the
same workload from `launchrunfarm`/`infrasetup` until it succeeds. Set a finite
limit with `--rvv-panic-retries=N` (use `0` to disable retries). Other guest or
FireSim failures still stop Stage 2 immediately.

Generated FireSim runtime configurations enable `host_debug.zero_out_dram` by
default so every workload starts from cleared target DRAM.

The paper's BOOM+Gemmini FlexAttention comparison uses OPT at sequence lengths
256, 512, 768, and 1024. Its 1-core workloads run only the first compiled
FlexAttention launch for Flash and Window attention, repeated five times. They
do not load the approximately 500 MiB OPT weight blob or execute the complete
transformer. The paper-default Stage 1 flow no longer creates unrelated Pythia
1-core BOOM workloads.

This kernel-only mode is part of the generated `model-1core.elf`. Existing OPT
ELFs created before this change must be regenerated before Stage 2; packaging
rejects an old ELF instead of silently running the full model. Running Stage 1
and then Stage 2 through the replication flow above regenerates the artifacts,
core-specific ELFs, workload packages, and FireMarshal images.

Stage 2 automatically packages the ablation as
`<model>-gemmini-sdpa-256tok-alias-first-{on,off}-4core` and maps both variants
to the fp32x8x8 Gemmini Rocket 4-core hardware configuration. CNN OFF cases are
packaged as `<model>-gemmini-alias-first-off-4core`; their existing ordinary
`<model>-gemmini-4core` results supply the ON values. Use `--only-alias-first`
to build and execute the four CNN OFF and eight LLM ON/OFF workloads. The
plotting stage writes `alias_first_ablation.csv` and `fig6c.{pdf,png}` using
alias-first OFF as the normalized 1.0
baseline. To regenerate only this panel, run `scripts/run-plot.sh
--only-alias-first`. If the LLM runs are already complete, use
`--only-alias-first-cnn-off` with Stage 1 and Stage 2 to generate and run only
the four missing CNN OFF cases.

## Citation

## Saturn RVV Kernel Workaround

The FireMarshal Linux kernel used by the artifact has local workarounds in
`$CHIPYARD_DIR/software/firemarshal/boards/default/linux/arch/riscv/kernel/traps.c`
and `arch/riscv/include/asm/vector.h`. The `jseo` comment in
`do_trap_ecall_u()` disables `riscv_v_vstate_discard(regs)` because the Saturn
target can intermittently lose `sstatus.VS` while running the LMUL=8 syscall
discard sequence and raise an illegal-instruction exception at
`vmv.v.i v24, -1`.

The artifact also leaves vector state resident on each hart instead of running
the LMUL=8 vector save/restore sequence during task switches. This AE-only
policy relies on the generated RVV workload pinning one userspace RVV thread to
each hart. Do not run another userspace RVV task on those harts or allow an RVV
workload thread to migrate between harts.

This is a Linux kernel change, so every FireMarshal image that runs an RVV
workload must be rebuilt. Stage 2 performs that rebuild. It does not require
rebuilding Stage 1 artifacts, `model-<N>core.elf`, FPGA bitstreams, or FireSim
driver bundles separately.

After rebuilding, confirm that the Linux build number in each RVV workload's
UART log differs from the pre-workaround build (`#126`).
