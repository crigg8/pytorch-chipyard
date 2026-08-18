# RVV c640ae2f02ebc584 verification matrix

The model/core-specific ELF is the fixed variable. `c640ae2f02ebc584` names
the artifact set; it is not a shared ELF hash. A workload counts as verified
only when its packaged ELF SHA256 matches its own corresponding entry below.
OpenMP, FireSim hardware, and runtime settings may vary, but the successful
combination must be recorded.

## Fixed ELF SHA256

| Workload | SHA256 |
| --- | --- |
| alexnet-rvv-2core | `ab34ae9cb2233d936b4ea8861edaf26e3721e52ca2fb4f4cbbdea6ea58a62ab3` |
| alexnet-rvv-4core | `7c023928fd673f782252e94b42bba04baeb2d56798fc4b9f5003047ed08f8318` |
| mobilenetv2-rvv-2core | `e25ea2aa04d0fa9d6ce21535e5920759e2a85159d1496246b78162922a1e0f57` |
| mobilenetv2-rvv-4core | `15f818605173b01f0f418d0d2e152c086eec7121a31bb1ae2b5cd343fb9b5755` |
| resnet50-rvv-2core | `423a1d41492ccfee8d23ed5f18fd191d3d1d6ed261ca14c626a5b23eb7828143` |
| resnet50-rvv-4core | `2def4511d598c75bc7b0fd6d2e60601958fdb2aae8d01facd2fa360f95ba139e` |
| squeezenet-rvv-2core | `08f62f9234604824c1521c82c1085f6eef5fa4406a39fa4697d421d28abcd6ba` |
| squeezenet-rvv-4core | `887fd2f264f6ec81563dde102043a317afd0d5ce77ff7d68c76724a7fc9bd40c` |

## Common successful settings

```text
OMP_STACKSIZE=64M
OMP_DYNAMIC=false
OMP_MAX_ACTIVE_LEVELS=1
OMP_NESTED=false
OMP_PROC_BIND=close
OMP_SCHEDULE=static
OMP_WAIT_POLICY=ACTIVE
OMP_DISPLAY_ENV=FALSE
GOMP_SPINCOUNT=INFINITE
MALLOC_ARENA_MAX=1

FPGA DB=/opt/firesim-db0.json
topology=no_net_config
link_latency=6405
switching_latency=10
net_bandwidth=200
profile_interval=-1
plusarg_passthrough=""
tracing.enable=no
autocounter.read_rate=0
host_debug.zero_out_dram=no
host_debug.disable_synth_asserts=no
terminate_on_completion=yes
```

## Verified combinations (8/8)

| Workload | Hardware | Worker placement | Other OMP | Evidence |
| --- | --- | --- | --- | --- |
| alexnet-rvv-2core | `MINV128D64Rocket2Config` | `OMP_PLACES={0},{1}`; `GOMP_CPU_AFFINITY=0 1` | `OMP_NUM_THREADS=2`; `OMP_THREAD_LIMIT=2`; `OMP_DISPLAY_AFFINITY=FALSE` | `results-workload/2026-08-17--06-51-47-alexnet-rvv-2core-alexnet-rvv-2core` |
| alexnet-rvv-4core | `MINV128D64Rocket8Config`, 30 MHz | `OMP_PLACES={4},{5},{6},{7}`; `GOMP_CPU_AFFINITY=4 5 6 7` | `OMP_NUM_THREADS=4`; `OMP_THREAD_LIMIT=4`; `OMP_DISPLAY_AFFINITY=TRUE` | `results-workload/2026-08-17--13-18-03-alexnet-rvv-4core-alexnet-rvv-4core` |
| mobilenetv2-rvv-2core | `MINV128D64Rocket4Config`, 30 MHz | `OMP_PLACES={2},{3}`; `GOMP_CPU_AFFINITY=2 3` | `OMP_NUM_THREADS=2`; `OMP_THREAD_LIMIT=2`; `OMP_DISPLAY_AFFINITY=FALSE` | `results-workload/2026-08-18--02-27-14-mobilenetv2-rvv-2core-mobilenetv2-rvv-2core` |
| mobilenetv2-rvv-4core | `MINV128D64Rocket8Config`, 30 MHz | `OMP_PLACES={0},{1},{2},{3}`; `GOMP_CPU_AFFINITY=0 1 2 3` | `OMP_NUM_THREADS=4`; `OMP_THREAD_LIMIT=4`; `OMP_DISPLAY_AFFINITY=TRUE` | `results-workload/2026-08-17--17-22-45-mobilenetv2-rvv-4core-mobilenetv2-rvv-4core` |
| resnet50-rvv-2core | `MINV128D64Rocket2Config` | `OMP_PLACES={0},{1}`; `GOMP_CPU_AFFINITY=0 1` | `OMP_NUM_THREADS=2`; `OMP_THREAD_LIMIT=2`; `OMP_DISPLAY_AFFINITY=FALSE` | `results-workload/2026-08-17--18-29-33-resnet50-rvv-2core-resnet50-rvv-2core` |
| resnet50-rvv-4core | `MINV128D64Rocket8Config`, 30 MHz | `OMP_PLACES={4},{5},{6},{7}`; `GOMP_CPU_AFFINITY=4 5 6 7` | `OMP_NUM_THREADS=4`; `OMP_THREAD_LIMIT=4`; `OMP_DISPLAY_AFFINITY=TRUE` | `results-workload/2026-08-17--14-43-42-resnet50-rvv-4core-resnet50-rvv-4core` |
| squeezenet-rvv-2core | `MINV128D64Rocket2Config` | `OMP_PLACES={0},{1}`; `GOMP_CPU_AFFINITY=0 1` | `OMP_NUM_THREADS=2`; `OMP_THREAD_LIMIT=2`; `OMP_DISPLAY_AFFINITY=FALSE` | `results-workload/2026-08-17--20-33-39-squeezenet-rvv-2core-squeezenet-rvv-2core` |
| squeezenet-rvv-4core | `MINV128D64Rocket8Config`, 30 MHz | `OMP_PLACES={4},{5},{6},{7}`; `GOMP_CPU_AFFINITY=4 5 6 7` | `OMP_NUM_THREADS=4`; `OMP_THREAD_LIMIT=4`; `OMP_DISPLAY_AFFINITY=TRUE` | `results-workload/2026-08-17--20-53-16-squeezenet-rvv-4core-squeezenet-rvv-4core` |

Hardware artifact hashes:

```text
MINV128D64Rocket2Config firesim.tar.gz:
  86e0d257e2c7def7a8ebe32f4033639cd20c431e8399047e53d0c9c5bbf006af
MINV128D64Rocket2Config driver-bundle.tar.gz:
  b935683df98c9a4122ff074e2f7ed86fe969599f9f91138aeba74e86c135968f
MINV128D64Rocket4Config 30 MHz firesim.tar.gz:
  de3fd57f104e26b5823f84b157a1f4c7c55c1ac574728fcb402e5abcdde379cf
MINV128D64Rocket4Config 30 MHz driver-bundle.tar.gz:
  1cef912186020aedd7148650aa06236f0a178bb55ff3ea96b6f71049e0677c5e
MINV128D64Rocket8Config 30 MHz firesim.tar.gz:
  0b9258aeebb4dfd69d1d75838dba79fc23d5c675a630d426470fa549ca208050
MINV128D64Rocket8Config 30 MHz driver-bundle.tar.gz:
  129139697ea95cf3bc96d2275cac9ebb3a08838155338c4b7d17fe897a216ee6
```

## Completion status

All eight model/core-specific fixed ELFs have a successful FireSim run with a
recorded hardware and OpenMP combination. There are no pending RVV CNN
workloads in this matrix.

The defaults are encoded in `scripts/package-firemarshal-workload.sh` and
`scripts/run-firesim-workloads.sh`: MobileNetV2 2core selects the physical
four-hart 30 MHz target and harts 2-3; every four-thread RVV workload selects
the physical eight-hart 30 MHz target, with MobileNetV2 on harts 0-3 and the
other CNNs on harts 4-7. Other two-thread RVV CNNs select the physical
two-hart target and harts 0-1. Per-workload environment overrides remain
available for diagnostic runs.

The 2026-08-16 `resnet50-rvv-2core` success is historical evidence only. It
used an earlier ELF and `OMP_WAIT_POLICY=PASSIVE`, `GOMP_SPINCOUNT=0`, harts
0-1, on `MINV128D64Rocket2Config`; 18 failed attempts preceded the success.
It therefore does not verify the fixed ELF above.

The fixed `mobilenetv2-rvv-2core` ELF failed on the physical two-hart target
with workers on harts 0-1, then passed on the physical four-hart 30 MHz target
with workers isolated to harts 2-3. The `mobilenetv2-rvv-4core` success used
the eight-hart 30 MHz target with workers on harts 0-3.
