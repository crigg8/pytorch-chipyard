# Plot input audit (2026-08-18)

## RVV CNN result

The eight fixed-ELF RVV CNN `model.log` and `autotune.log` pairs were copied
into `scripts/figures/results-workload/<workload>/`. No RVV rerun is required
for the Saturn/RVV CNN plots.

All eight logs contain a parseable `Avg Model cycle`, `Model samples: 5`, an
execution-order kernel table, and autotune metadata:

| Workload | Execution rows | Autotune kernels |
| --- | ---: | ---: |
| alexnet-rvv-2core | 21 | 73 |
| alexnet-rvv-4core | 21 | 73 |
| mobilenetv2-rvv-2core | 56 | 343 |
| mobilenetv2-rvv-4core | 56 | 343 |
| resnet50-rvv-2core | 74 | 332 |
| resnet50-rvv-4core | 74 | 332 |
| squeezenet-rvv-2core | 63 | 146 |
| squeezenet-rvv-4core | 63 | 146 |

The isolated plot test generated `cnn_result_saturn.pdf` (Figure 7(b)) with all four models
and both 2-thread and 4-thread bars.

Two input-selection bugs were fixed in `generate_plot_inputs.py`:

- diagnostic CNN variants such as `mobilenetv2-rvv-nohazard-2core` no longer
  overwrite the canonical RVV baseline cell;
- alias-first attention ablations no longer stand in for missing canonical
  SDPA/Flash/Window runs.

## Whole plot suite

The whole suite is not yet reproducible from the currently collected results.
This is missing experiment data, not a failure to parse the new RVV logs.

- `cnn_result_saturn.pdf` (Figure 7(b)): complete.
- `alias_first_ablation` and `autotune_gemmini_max`: complete.
- Rocket CNN scaling: missing ResNet50 and SqueezeNet scalar/Rocket 4-, 8-,
  and 16-core logs.
- Absolute CNN cycles and Gemmini scaling: also missing ResNet50 and
  SqueezeNet Gemmini 2-core logs.
- im2col figures: missing ResNet50 and SqueezeNet Gemmini-im2col 4-core logs.
- SDPA 256 figure: currently contains GPT-2 and GPT-Neo only; OPT and Pythia
  canonical 2-/4-core runs are absent.
- Flex-prefill and Flash/Window ratio figures: canonical SDPA/Flash/Window
  sweeps are absent. Alias-first logs are intentionally no longer reused.
- MobileNet/SqueezeNet attribution: missing SqueezeNet scalar/Rocket 4-core
  and Gemmini 2-core logs.

Therefore, do not rerun all workloads. Keep the current RVV results. If every
paper figure must be regenerated, run only the missing non-RVV workloads above
and then rerun `scripts/figure/plot_results.sh`.
