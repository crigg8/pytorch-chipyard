# pytorch-chipyard
A Reusable Compiler-Stack Foundation for ML Research on Chipyard Hardware.

이 readme에서는 논문의 evaluation을 replicate 하는 방법에 대한 설명에 초점을 맞추고 있다. 더욱 자세한 설명은 document를 참고하도록 해라.
https://pytorch-chipyard.readthedocs.io/en/latest/


## Installation

```
# pre-requisite: conda-24.11.3
git clone https://github.com/JongseoKang/pytorch-chipyard

cd ./pytorch-chipyard
git submodule update --init pytorch triton triton_chipyard llvm-project buddy-mlir

# install others
bash scripts/install.sh
```

### Local FPGA host prerequisite

The installation commands above initialize this repository's compiler stack.
They do not install Chipyard/FireSim or configure the Linux host as a FireSim
FPGA run farm machine. Local FPGA support is a separate machine-level
prerequisite because it installs privileged helper scripts under `/usr/local/bin`
and configures device access for the FPGA host.

On the authors' artifact review server, this prerequisite is already handled.
On a self-managed U250 host, install the FireSim helper scripts from the local
Chipyard/FireSim checkout that will be used for Stage 2:

```bash
cd /path/to/local/chipyard/sims/firesim
sudo cp deploy/sudo-scripts/* /usr/local/bin
sudo cp platforms/xilinx_alveo_u250/scripts/* /usr/local/bin
sudo chmod 755 /usr/local/bin/firesim*
```

The host also needs the corresponding FireSim group/sudoers setup described by
FireSim's local FPGA setup guide, for example allowing the `firesim` group to
run `/usr/local/bin/firesim-*` without a password. These scripts are
host-global files, so they are intentionally not installed by
`scripts/install.sh`.

The evaluated host setup used FireSim commit
`141bff735c9b81a6d82a593310d0ca8be903e9a1`, which is on FireSim `main` after
the `1.20.1` release. Use the FireSim `latest` local FPGA setup document for the
rendered guide:
[FireSim Local FPGA Initial Setup](https://docs.fires.im/en/latest/Local-FPGA-Initial-Setup.html).
For an exact source permalink matching this checkout, see
[docs/Local-FPGA-Initial-Setup.rst at 141bff735](https://github.com/firesim/firesim/blob/141bff735c9b81a6d82a593310d0ca8be903e9a1/docs/Local-FPGA-Initial-Setup.rst).

## Artifact Replication

The default path for regenerating the paper figures is:

1. Prepare the FPGA host as described in the installation prerequisite above.
2. Point `scripts/env.sh` at the local Chipyard/FireSim checkout:

   ```bash
   export CHIPYARD_DIR=/path/to/local/chipyard
   # Optional if the host does not use /opt/firesim-db.json:
   # export PYTORCH_CHIPYARD_FPGA_DB=/path/to/firesim-db.json
   # Optional if hwdb/build recipes live outside $CHIPYARD_DIR/sims/firesim/deploy:
   # export FIRESIM_HWDB_PATH=/path/to/config_hwdb.yaml
   # export FIRESIM_BUILD_RECIPES_PATH=/path/to/config_build_recipes.yaml
   source scripts/env.sh
   ```

3. Download the released U250 bitstream artifacts from the GitHub release and
   place them under FireSim's expected build-result tree:

   ```text
   $FIRESIM_DEPLOY_DIR/results-build/
   ```

   The `config_hwdb.yaml` entries used by FireSim should resolve to the
   downloaded `firesim.tar.gz` files under this directory. If a host stores
   bitstreams somewhere else, update
   `$FIRESIM_HWDB_PATH` before running FireSim.

4. From the repository root, run the full default artifact-generation and FPGA
   reproduction flow. Stage 1 can run in the compiler Docker environment; Stage
   2 should run on the local Chipyard/FireSim FPGA host.

   ```bash
   # Stage 1: generate the paper-default compiler artifacts under examples/
   bash scripts/run-cnn.sh
   bash scripts/run-sdpa.sh
   bash scripts/run-im2col.sh
   bash scripts/run_gemmini_autotune.sh
   bash scripts/run-flex-attn.sh

   # Stage 2: build ELFs, package, build images, run FireSim, and generate figures.
   # Run these on the local Chipyard/FireSim host after the env setup above.
   bash scripts/build-chipyard-elves.sh --artifact-root=examples
   bash scripts/package-firemarshal-workload.sh
   bash scripts/build-firemarshal-images.sh
   bash scripts/run-firesim-workloads.sh
   bash scripts/figure/plot_results.sh
   ```

The final figures are written under `scripts/figures/`.

With no options, the Stage 1 scripts expand to the paper-required default
workload sets and write artifacts under `examples/`. This full path is
expensive: long-sequence LLM FPGA runs can take hours per workload. The sections
below describe what each step does and how to override paths or run subsets of
the workload set.

The full workflow is organized in two stages:

1. Compile PyTorch models into portable compiler artifacts.
2. Build those artifacts into RISC-V ELFs, then package and run them through
   FireMarshal/FireSim on an FPGA host.

Full replication requires an AMD/Xilinx Alveo U250. Some experiments may run on
an Alveo U280 with matching bitstreams(some experiments require large bitstream), but the full paper configuration assumes U250 bitstreams.

This repository contains the compiler stack. It does not install Chipyard,
FireMarshal, FireSim, or configure the FPGA host system. The host must already
provide a working FPGA runtime environment, including XRT, XDMA,
Vivado/Vitis-compatible tooling, FireSim database files, and access to the
target FPGA device.

Reference host configuration checked on the authors' FPGA server:

- FPGA: AMD/Xilinx Alveo U250-class FireSim host
- OS: Ubuntu 20.04.6 LTS (Focal Fossa)
- Kernel: Linux 5.4.0-216-generic
- CPU/memory: 80 CPU cores, 192063 MB memory
- XRT: 2.16.204, branch 2023.2
- XDMA: kernel module loaded, version 2020.2.2
- Vivado: v2021.1, SW Build 3247384

Prebuilt bitstreams are distributed through the GitHub release artifacts.
Using the released bitstreams is recommended for performance replication because
Chipyard and FireSim hardware configurations can change across builds.

The artifact workflow has two stages. The first stage is Torch model
compilation. The detailed commands below are run from the repository root when
you want to regenerate only a subset or use a non-default artifact directory.
Arguments are passed as named options so that individual options can be omitted
without making the command ambiguous. When an option is omitted or set to
`default`, the script runs the paper-required default set for that experiment.
Options such as `--model`, `--backend`, `--attention`, and `--seq-len` may also
take comma-separated lists.

### Stage 1: Torch model compilation

#### CNN workloads

CNN experiments cover `AlexNet`, `MobileNetV2`, `ResNet50`, and `SqueezeNet`.

```bash
# Usage:
bash scripts/run-cnn.sh \
  --backend=[gemmini|rvv|scalar|default|comma-separated-list] \
  --model=[alexnet|mobilenetv2|resnet50|squeezenet|default|comma-separated-list] \
  --artifact-dir=<path>

# Example:
bash scripts/run-cnn.sh --backend=gemmini --model=mobilenetv2

# Example with multiple models/backends:
bash scripts/run-cnn.sh --backend=gemmini,rvv --model=alexnet,resnet50
```

If the artifact directory is omitted, generated files are placed under
`examples/<model>/<backend>/`. For example, the Gemmini MobileNetV2 run should
produce paper-required executables such as:

```text
examples/mobilenetv2/gemmini/model-2core.elf
examples/mobilenetv2/gemmini/model-4core.elf
examples/mobilenetv2/gemmini/input.bin
examples/mobilenetv2/gemmini/weights.bin
```

#### LLM SDPA prefill workloads

The SDPA prefill experiments cover `gpt2`, `gpt-neo`, `opt`, and `pythia`. The
paper configuration uses Gemmini 2-core and 4-core targets with sequence length
256.

```bash
# Usage:
bash scripts/run-sdpa.sh \
  --model=[gpt2|gpt-neo|opt|pythia|default|comma-separated-list] \
  --artifact-dir=<path>

# Example:
bash scripts/run-sdpa.sh --model=opt

# Example with multiple models:
bash scripts/run-sdpa.sh --model=opt,pythia
```

#### FX lowering experiment

The FX lowering experiment compares the normal direct convolution lowering with
the im2col+matmul lowering path.

```bash
# Usage:
bash scripts/run-im2col.sh \
  --backend=[gemmini|rvv|scalar|default|comma-separated-list] \
  --model=[alexnet|mobilenetv2|resnet50|squeezenet|default|comma-separated-list] \
  --artifact-dir=<path>

# Example:
bash scripts/run-im2col.sh --backend=gemmini --model=resnet50
```

The direct convolution baseline is generated by `scripts/run-cnn.sh`. The
im2col script is responsible for enabling the TorchInductor im2col lowering path
and writing its artifacts to a separate directory so that the two variants can be
compared without overwriting each other.

#### Gemmini autotuning experiment

The Gemmini autotuning experiment uses the paper GEMM shape:

```text
(1024 x 1024) x (1024 x 4096)
```

```bash
bash scripts/run_gemmini_autotune.sh --artifact-dir=<path>
```

This run should enable the Gemmini max-autotune path and generate a workload
containing only the target matrix multiplication. The generated artifacts must
include `autotune.log`.

#### FlexAttention experiments

FlexAttention experiments cover `opt` and `pythia`, attention modes `sdpa`,
`flash`, and `window`, and sequence lengths 256, 512, 768, and 1024.

```bash
# Usage:
bash scripts/run-flex-attn.sh \
  --model=[opt|pythia|default|comma-separated-list] \
  --attention=[sdpa|flash|window|default|comma-separated-list] \
  --host=[rocket|boom|default|comma-separated-list] \
  --seq-len=[256|512|768|1024|default|comma-separated-list] \
  --artifact-dir=<path>

# Example:
bash scripts/run-flex-attn.sh \
  --model=opt \
  --attention=window \
  --host=rocket \
  --seq-len=1024

# Example with multiple models and sequence lengths:
bash scripts/run-flex-attn.sh \
  --model=opt,pythia \
  --attention=flash,window \
  --host=rocket \
  --seq-len=256,512,768,1024
```

While the Rocket+Gemmini experiments use the 4-core target, the BOOM+Gemmini window-vs-flash comparison uses a 1-core software configuration. 
This is due to BOOM's out-of-order execution, parallelization via OpenMP does not work well.
Long sequence lengths are expensive; on the authors' setup, a single sequence-1024 model run took roughly five hours.

### Stage 2: FPGA simulation

The second stage uses a local Chipyard/FireSim host setup to build Stage 1
compiler artifacts into RISC-V ELFs, package those ELFs into FireMarshal
workloads, build/install FireMarshal images, and run FireSim. The compiler
artifact repository does not need to contain a `chipyard/` checkout, but the
host running this stage must provide initialized Chipyard, FireMarshal, and
FireSim paths.

Before running this stage, point `scripts/env.sh` at the local host setup.
For the standard Chipyard layout, `CHIPYARD_DIR` is the only required user input;
the other paths are derived by `source scripts/env.sh`:

```bash
export CHIPYARD_DIR=/path/to/local/chipyard
# Optional if the host does not use /opt/firesim-db.json:
# export PYTORCH_CHIPYARD_FPGA_DB=/path/to/firesim-db.json
# Optional if hwdb/build recipes live outside $CHIPYARD_DIR/sims/firesim/deploy:
# export FIRESIM_HWDB_PATH=/path/to/config_hwdb.yaml
# export FIRESIM_BUILD_RECIPES_PATH=/path/to/config_build_recipes.yaml
source scripts/env.sh
```

`source scripts/env.sh` is for populating the current shell with derived paths
such as `CHIPYARD_ENV_PATH`, `FIREMARSHAL_DIR`, `FIRESIM_DIR`,
`FIRESIM_DEPLOY_DIR`, `FIRESIM_WORKLOAD_DIR`,
`PYTORCH_CHIPYARD_WORKLOAD_DIR`, and
`PYTORCH_CHIPYARD_RESULTS_WORKLOAD_DIR`. Run the workflow scripts themselves
with `bash`; they source `scripts/env.sh` internally and may call `exit` or
install traps.

The FireMarshal image build step mounts rootfs images while applying overlays,
so the user running the script is expected to have `sudo` permission.

#### Local Chipyard ELF build

Build `model-<N>core.elf` files from the generated Stage 1 artifacts:

```bash
bash scripts/build-chipyard-elves.sh --artifact-root=examples
```

If Stage 1 was run with a custom artifact root, pass that same path with
`--artifact-root`.

#### FireMarshal workload/package creation

From the repository root, package every generated Stage 1 artifact:

```bash
bash scripts/package-firemarshal-workload.sh
```

By default the script scans `examples/` for `model-<N>core.elf` files and
packages all discovered workloads. If Stage 1 was run with a custom
`--artifact-dir`, pass the same directory:

```bash
bash scripts/package-firemarshal-workload.sh --artifact-root=<artifact-dir>
```

This writes FireMarshal workload JSON/overlays under:

```text
$PYTORCH_CHIPYARD_WORKLOAD_DIR/
```

and FireSim workload JSON files under:

```text
$FIRESIM_WORKLOAD_DIR/
```

The generated FireMarshal workloads inherit directly from FireMarshal's
`br-base.json`. For LLM workloads with sequence length greater than 256, the
generated guest runner passes `--no-output` to the ELF to avoid writing very
large output tensors.

#### FireMarshal image build/install

Build all packaged FireMarshal images and install the corresponding FireSim
workloads:

```bash
bash scripts/build-firemarshal-images.sh
```

The script runs `sudo -v` before invoking FireMarshal and keeps the sudo
credential alive while images are being built. Internally it runs the equivalent
of:

```bash
cd "$FIREMARSHAL_DIR"
./marshal build <generated-workload-jsons>
./marshal install <generated-workload-jsons>
```

Expected outputs:

```text
$FIREMARSHAL_IMAGE_DIR/<workload>/<workload>-bin
$FIREMARSHAL_IMAGE_DIR/<workload>/<workload>.img
$FIRESIM_WORKLOAD_DIR/<workload>.json
```

This step does not consume FPGA bitstreams directly. Bitstreams are selected in
the FireSim run stage through `$FIRESIM_HWDB_PATH` and
`$FIRESIM_BUILD_RECIPES_PATH`, which should point to the released U250 bitstream
artifacts on the FPGA host.

#### FireSim execution

Run every packaged workload on the FPGA and collect the latest results:

```bash
bash scripts/run-firesim-workloads.sh
```

The script discovers workloads from:

```text
$PYTORCH_CHIPYARD_WORKLOAD_DIR/
```

For each workload, it generates a per-workload FireSim runtime config under
`.firesim-runtime/` and runs:

```bash
firesim launchrunfarm
firesim infrasetup
firesim runworkload
firesim terminaterunfarm
```

It does not edit FireSim's checked-in `config_runtime.yaml`. The generated
runtime configs use the paths exported by `scripts/env.sh`, including
`FIRESIM_RUNS_DIR`, `PYTORCH_CHIPYARD_FPGA_DB`, `FIRESIM_HWDB_PATH`, and
`FIRESIM_BUILD_RECIPES_PATH`. By default, `PYTORCH_CHIPYARD_FPGA_DB` points to
`/opt/firesim-db.json`; override it if the host uses a different FPGA database.

To run only selected workloads:

```bash
bash scripts/run-firesim-workloads.sh --workload=resnet50-rvv-2core
bash scripts/run-firesim-workloads.sh --workload=resnet50-rvv-2core,opt-gemmini-window-256tok-4core
```

The hardware config is inferred from the workload name. If a workload needs a
specific bitstream entry, override it with either a per-workload variable:

```bash
PYTORCH_CHIPYARD_FIRESIM_HW_CONFIG_RESNET50_GEMMINI_4CORE=alveo_u250_firesim_fp8x8_gemmini_rocket_4core_no_nic \
  bash scripts/run-firesim-workloads.sh --workload=resnet50-gemmini-4core
```

or a comma-separated mapping:

```bash
PYTORCH_CHIPYARD_FIRESIM_HW_CONFIG_OVERRIDES=resnet50-gemmini-4core=alveo_u250_firesim_fp8x8_gemmini_rocket_4core_no_nic \
  bash scripts/run-firesim-workloads.sh --workload=resnet50-gemmini-4core
```

After each run, FireSim creates a timestamped directory under:

```text
$PYTORCH_CHIPYARD_RESULTS_WORKLOAD_DIR/
```

The script copies the latest result files into timestamp-free workload folders:

```text
scripts/figures/results-workload/<workload>/model.log
scripts/figures/results-workload/<workload>/autotune.log
scripts/figures/results-workload/<workload>/output.bin
```

For LLM workloads with sequence length greater than 256, `output.bin` is not
generated or collected because the FireMarshal runner uses `--no-output`.
Existing FireSim results can be collected without running the FPGA again:

```bash
bash scripts/run-firesim-workloads.sh --collect-only
```

#### Figure generation

After the FireSim results have been collected, generate the paper figures:

```bash
bash scripts/figure/plot_results.sh
```

The script regenerates the plot CSV/log inputs from
`scripts/figures/results-workload/`, runs all figure plotting scripts under
`scripts/figure/`, and prints the generated figure paths at the end. It does
not run validation or generate paper tables.

If the FireSim results were collected somewhere else, pass that directory:

```bash
bash scripts/figure/plot_results.sh --results-dir=<results-workload-dir>
```

The default Python environment is the `pytorch-chipyard` conda environment
created by `scripts/install.sh`; set `PYTHON_BIN=/path/to/python` to use a
different environment with `matplotlib`, `numpy`, and `pandas` installed.
