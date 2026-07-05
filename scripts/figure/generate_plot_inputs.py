from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = ROOT_DIR.parent
DEFAULT_RESULTS_DIR = ROOT_DIR / "figures" / "results-workload"
CSV_DIR = ROOT_DIR / ".csv"
LOG_DIR = Path(os.environ.get("PYTORCH_CHIPYARD_LOG_DIR", WORKSPACE_DIR / "examples" / ".logs"))
GENERATED_CSVS = [
    "cnn_result.csv",
    "spda_prefill_256.csv",
    "flex_prefill.csv",
    "flash_window_core_ratio.csv",
    "im2col_site_attribution.csv",
]

CNN_MODELS = ["ResNet", "AlexNet", "MobileNet", "SqueezeNet"]
CNN_MODEL_ALIASES = {
    "alexnet": "AlexNet",
    "mobilenet": "MobileNet",
    "mobilenetv2": "MobileNet",
    "resnet": "ResNet",
    "resnet50": "ResNet",
    "squeezenet": "SqueezeNet",
}
LLM_MODELS = {"gpt2", "gpt-neo", "opt", "pythia"}
MODEL_PREFIXES = sorted(
    list(CNN_MODEL_ALIASES) + list(LLM_MODELS),
    key=len,
    reverse=True,
)


@dataclass(frozen=True)
class WorkloadRun:
    workload: str
    result_dir: Path
    model_log: Path
    autotune_log: Path | None
    avg_cycles: float
    samples: int
    model: str
    tags: tuple[str, ...]
    core: int
    tokens: int | None

    @property
    def is_llm(self) -> bool:
        return self.model in LLM_MODELS or self.tokens is not None

    @property
    def is_cnn(self) -> bool:
        return self.model in CNN_MODEL_ALIASES


def log(message: str) -> None:
    print(f"[plot-inputs] {message}")


def warn(message: str) -> None:
    print(f"[plot-inputs][warn] {message}")


def parse_model_log(path: Path) -> tuple[float, int]:
    text = path.read_text(errors="replace")
    avg_match = re.search(r"Avg Model cycle:\s*([0-9.]+)", text)
    samples_match = re.search(r"Model samples:\s*(\d+)", text)
    if avg_match is None:
        raise ValueError(f"missing Avg Model cycle in {path}")
    return float(avg_match.group(1)), int(samples_match.group(1)) if samples_match else 1


def parse_workload_name(workload: str) -> tuple[str, tuple[str, ...], int, int | None]:
    workload = workload.lower()
    core_match = re.search(r"(?:^|-)(\d+)core$", workload)
    if core_match is None:
        raise ValueError(f"could not infer core count from workload name: {workload}")
    core = int(core_match.group(1))
    stem = workload[: core_match.start()].rstrip("-")

    if stem.startswith("gemmini-max-autotune"):
        tags = tuple(part for part in stem.split("-") if part)
        return "gemmini-max-autotune", tags, core, None

    tokens = None
    token_match = re.search(r"(?:^|-)(\d+)tok(?:-|$)", stem)
    if token_match is not None:
        tokens = int(token_match.group(1))
        stem = (stem[: token_match.start()] + "-" + stem[token_match.end() :]).strip("-")

    for prefix in MODEL_PREFIXES:
        if stem == prefix:
            return prefix, tuple(), core, tokens
        if stem.startswith(f"{prefix}-"):
            tags = tuple(part for part in stem[len(prefix) + 1 :].split("-") if part)
            return prefix, tags, core, tokens

    raise ValueError(f"could not infer model from workload name: {workload}")


def workload_from_result_log(model_log: Path) -> str:
    parent = model_log.parent.name
    if parent.endswith("0") and len(parent) > 1:
        candidate = parent[:-1]
        if candidate in model_log.parent.parent.name:
            return candidate
    return parent


def discover_runs(results_dir: Path) -> list[WorkloadRun]:
    latest: dict[str, Path] = {}
    for model_log in results_dir.rglob("model.log"):
        workload = workload_from_result_log(model_log)
        previous = latest.get(workload)
        if previous is None or model_log.stat().st_mtime > previous.stat().st_mtime:
            latest[workload] = model_log

    runs: list[WorkloadRun] = []
    for workload, model_log in sorted(latest.items()):
        try:
            model, tags, core, tokens = parse_workload_name(workload)
            avg_cycles, samples = parse_model_log(model_log)
        except ValueError as exc:
            warn(str(exc))
            continue

        autotune_log = model_log.with_name("autotune.log")
        runs.append(
            WorkloadRun(
                workload=workload,
                result_dir=model_log.parent,
                model_log=model_log,
                autotune_log=autotune_log if autotune_log.exists() else None,
                avg_cycles=avg_cycles,
                samples=samples,
                model=model,
                tags=tags,
                core=core,
                tokens=tokens,
            )
        )
    return runs


def model_alias_workloads(workload: str) -> list[str]:
    aliases = {workload.lower()}
    changed = True
    while changed:
        changed = False
        for alias in list(aliases):
            candidates: list[str] = []
            if alias.startswith("mobilenetv2-"):
                candidates.append("mobilenet-" + alias[len("mobilenetv2-") :])
            elif alias.startswith("mobilenet-"):
                candidates.append("mobilenetv2-" + alias[len("mobilenet-") :])

            if alias.startswith("resnet50-"):
                candidates.append("resnet-" + alias[len("resnet50-") :])
            elif alias.startswith("resnet-"):
                candidates.append("resnet50-" + alias[len("resnet-") :])

            if "-scalar-" in alias:
                candidates.append(alias.replace("-scalar-", "-rocket-"))
            if "-rocket-" in alias:
                candidates.append(alias.replace("-rocket-", "-scalar-"))

            if alias.startswith("gemmini-max-autotune-gemmini-"):
                candidates.append(
                    "gemmini-max-autotune-fp32-"
                    + alias[len("gemmini-max-autotune-gemmini-") :]
                )
            elif alias.startswith("gemmini-max-autotune-fp32-"):
                candidates.append(
                    "gemmini-max-autotune-gemmini-"
                    + alias[len("gemmini-max-autotune-fp32-") :]
                )

            for candidate in candidates:
                if candidate not in aliases:
                    aliases.add(candidate)
                    changed = True
    return sorted(aliases)


def reset_generated_inputs() -> None:
    CSV_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    for csv_name in GENERATED_CSVS:
        path = CSV_DIR / csv_name
        if path.exists():
            path.unlink()
    for log_path in LOG_DIR.glob("*.log"):
        log_path.unlink()


def prepare_compat_logs(runs: list[WorkloadRun]) -> None:
    copied = 0
    for run in runs:
        for alias in model_alias_workloads(run.workload):
            shutil.copyfile(run.model_log, LOG_DIR / f"{alias}-model.log")
            copied += 1
            if run.autotune_log is not None:
                shutil.copyfile(run.autotune_log, LOG_DIR / f"{alias}-autotune.log")
                copied += 1
    log(f"wrote {copied} compatibility logs under {LOG_DIR}")


def has_tag(run: WorkloadRun, tag: str) -> bool:
    return tag in run.tags


def cnn_device_and_column(run: WorkloadRun) -> tuple[str, str] | None:
    if has_tag(run, "rvv"):
        if run.core == 2:
            return "Saturn", "2 Saturn"
        if run.core == 4:
            return "Saturn", "4 Saturn"
        return None

    if has_tag(run, "gemmini"):
        if run.core == 2:
            return "Gemmini", "Dual Gemmini"
        if run.core == 4:
            return "Gemmini", "Quad Gemmini"
        return None

    if has_tag(run, "scalar") or has_tag(run, "rocket"):
        if run.core in {4, 8, 16}:
            return "Rocket", f"{run.core}-core"
    return None


def write_cnn_result_csv(runs: list[WorkloadRun]) -> Path:
    columns = [
        "variant",
        "device",
        "model",
        "4-core",
        "8-core",
        "16-core",
        "2 Saturn",
        "4 Saturn",
        "Dual Gemmini",
        "Quad Gemmini",
    ]
    row_map: dict[tuple[str, str, str], dict[str, str]] = {}

    for variant in ["baseline", "im2col"]:
        for device in ["Rocket", "Saturn", "Gemmini"]:
            for model in CNN_MODELS:
                row_map[(variant, device, model)] = {
                    column: "" for column in columns
                } | {"variant": variant, "device": device, "model": model}

    for run in runs:
        if not run.is_cnn:
            continue
        model = CNN_MODEL_ALIASES[run.model]
        variant = "im2col" if has_tag(run, "im2col") else "baseline"
        device_column = cnn_device_and_column(run)
        if device_column is None:
            continue
        device, column = device_column
        row_map[(variant, device, model)][column] = f"{run.avg_cycles:.0f}"

    path = CSV_DIR / "cnn_result.csv"
    write_rows(path, columns, list(row_map.values()))
    return path


def best_run(runs: list[WorkloadRun], predicate) -> WorkloadRun | None:
    candidates = [run for run in runs if predicate(run)]
    if not candidates:
        return None

    def score(run: WorkloadRun) -> tuple[int, int, int]:
        return (
            1 if has_tag(run, "rocket") else 0,
            1 if not has_tag(run, "boom") else 0,
            run.core,
        )

    return sorted(candidates, key=score, reverse=True)[0]


def write_sdpa_csv(runs: list[WorkloadRun]) -> Path:
    columns = ["model", "variant", "cores", "total_kernel_cycles_avg"]
    rows: list[dict[str, str | int]] = []
    for model in ["opt", "pythia", "gpt2", "gpt-neo"]:
        for core in [2, 4]:
            run = best_run(
                runs,
                lambda item, model=model, core=core: item.model == model
                and item.tokens in {None, 256}
                and item.core == core
                and (has_tag(item, "sdpa") or not any(tag in item.tags for tag in ["flash", "window"])),
            )
            if run is None:
                continue
            rows.append(
                {
                    "model": model,
                    "variant": "fp32",
                    "cores": core,
                    "total_kernel_cycles_avg": f"{run.avg_cycles:.0f}",
                }
            )

    path = CSV_DIR / "spda_prefill_256.csv"
    write_rows(path, columns, rows)
    return path


def attention_run(
    runs: list[WorkloadRun],
    model: str,
    tokens: int,
    attention: str,
    host: str | None = None,
) -> WorkloadRun | None:
    def predicate(run: WorkloadRun) -> bool:
        if run.model != model or run.tokens != tokens or not has_tag(run, attention):
            return False
        if host == "boom":
            return has_tag(run, "boom")
        if host == "rocket":
            return not has_tag(run, "boom")
        return True

    return best_run(runs, predicate)


def write_flex_prefill_csv(runs: list[WorkloadRun]) -> Path:
    columns = ["model", "tokens", "sdpa_cycles", "flash_cycles", "window_cycles"]
    rows: list[dict[str, str | int]] = []
    for model in ["opt", "pythia"]:
        for tokens in [256, 512, 768, 1024]:
            row: dict[str, str | int] = {"model": model, "tokens": tokens}
            for attention, column in [
                ("sdpa", "sdpa_cycles"),
                ("flash", "flash_cycles"),
                ("window", "window_cycles"),
            ]:
                run = attention_run(runs, model, tokens, attention, host="rocket")
                row[column] = f"{run.avg_cycles:.0f}" if run else ""
            rows.append(row)

    path = CSV_DIR / "flex_prefill.csv"
    write_rows(path, columns, rows)
    return path


def write_flash_window_ratio_csv(runs: list[WorkloadRun]) -> Path:
    columns = ["model", "core", "tokens", "flash_window_ratio"]
    rows: list[dict[str, str | int]] = []
    for model in ["opt", "pythia"]:
        for core_label, host in [("Rocket", "rocket"), ("BOOM", "boom")]:
            for tokens in [256, 512, 768, 1024]:
                flash = attention_run(runs, model, tokens, "flash", host=host)
                window = attention_run(runs, model, tokens, "window", host=host)
                if flash is None or window is None or window.avg_cycles <= 0:
                    continue
                rows.append(
                    {
                        "model": model,
                        "core": core_label,
                        "tokens": tokens,
                        "flash_window_ratio": f"{flash.avg_cycles / window.avg_cycles:.6f}",
                    }
                )

    path = CSV_DIR / "flash_window_core_ratio.csv"
    write_rows(path, columns, rows)
    return path


def parse_autotune_kernel_metadata(path: Path | None) -> dict[str, dict[str, int | str]]:
    if path is None or not path.exists():
        return {}
    text = path.read_text(errors="replace")
    metadata: dict[str, dict[str, int | str]] = {}
    for block in text.split("\nAutotune Candidate\n")[1:]:
        kernel = extract(r"^Kernel:\s*(\S+)$", block)
        if not kernel:
            continue
        metadata[kernel] = {
            "raw": extract(r"^Raw kernel:\s*(\S+)$", block) or "",
            "kernel_h": int(extract(r"^\s+KERNEL_H=(\d+)$", block) or 0),
            "kernel_w": int(extract(r"^\s+KERNEL_W=(\d+)$", block) or 0),
            "stride_h": int(extract(r"^\s+STRIDE_H=(\d+)$", block) or 0),
            "padding_h": int(extract(r"^\s+PADDING_H=(\d+)$", block) or 0),
        }
    return metadata


def extract(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def parse_execution_kernel_rows(path: Path) -> list[tuple[str, int]]:
    text = path.read_text(errors="replace")
    if "All Kernel Cycle Stats (execution order)" not in text:
        return []
    stats = text.split("All Kernel Cycle Stats (execution order)", 1)[1]
    rows: list[tuple[str, int]] = []
    for block in re.split(r"\n(?=\d+\.\s+)", stats):
        name = extract(r"^\d+\.\s+(\S+)$", block)
        total = extract(r"^Total launch cycle:\s*(\d+)", block)
        if name and total:
            rows.append((name, int(total)))
    return rows


def im2col_label(model: str, kernel: str, metadata: dict[str, int | str] | None) -> str:
    raw = str(metadata.get("raw", "")) if metadata else ""
    kh = int(metadata.get("kernel_h", 0)) if metadata else 0
    sh = int(metadata.get("stride_h", 0)) if metadata else 0
    ph = int(metadata.get("padding_h", 0)) if metadata else 0

    if "prepack" in kernel or "prepack" in raw:
        return "prepack"
    if "convolution" not in raw and "convolution" not in kernel and "mm" not in raw:
        return "others"
    if model == "ResNet50":
        if kh == 7:
            return "7x7 conv"
        if kh == 3:
            return "3x3 conv"
        if kh == 1:
            return "1x1 conv"
    if model == "SqueezeNet":
        if kh == 3 and sh == 2 and ph == 0:
            return "3x3 s2p0"
        if kh == 3:
            return "Fire 3x3"
        if kh == 1:
            return "1x1 conv"
    return "others"


def cycles_by_im2col_label(run: WorkloadRun, labels: list[str]) -> dict[str, float]:
    totals = {label: 0.0 for label in labels}
    metadata = parse_autotune_kernel_metadata(run.autotune_log)
    model_label = "ResNet50" if run.model in {"resnet", "resnet50"} else "SqueezeNet"
    for kernel, total_cycles in parse_execution_kernel_rows(run.model_log):
        label = im2col_label(model_label, kernel, metadata.get(kernel))
        if label not in totals:
            label = "others"
        totals[label] += total_cycles / run.samples / 1e6
    return totals


def find_cnn_run(
    runs: list[WorkloadRun],
    model_names: set[str],
    required_tags: set[str],
    forbidden_tags: set[str] | None = None,
    core: int = 4,
) -> WorkloadRun | None:
    forbidden_tags = forbidden_tags or set()
    candidates = [
        run
        for run in runs
        if run.model in model_names
        and run.core == core
        and required_tags.issubset(set(run.tags))
        and not forbidden_tags.intersection(run.tags)
    ]
    return sorted(candidates, key=lambda run: run.avg_cycles)[0] if candidates else None


def write_im2col_site_csv(runs: list[WorkloadRun]) -> Path:
    columns = ["model", "label", "direct_mcycles", "im2col_mcycles", "delta_mcycles", "kind"]
    configs = [
        (
            "ResNet50",
            {"resnet", "resnet50"},
            ["7x7 conv", "1x1 conv", "3x3 conv", "prepack", "others"],
        ),
        (
            "SqueezeNet",
            {"squeezenet"},
            ["3x3 s2p0", "Fire 3x3", "1x1 conv", "prepack", "others"],
        ),
    ]
    rows: list[dict[str, str]] = []

    for model_label, model_names, labels in configs:
        direct = find_cnn_run(runs, model_names, {"gemmini"}, {"im2col"}, core=4)
        im2col = find_cnn_run(runs, model_names, {"gemmini", "im2col"}, core=4)
        if direct is None or im2col is None:
            continue

        direct_totals = cycles_by_im2col_label(direct, labels)
        im2col_totals = cycles_by_im2col_label(im2col, labels)
        for label in labels:
            direct_value = direct_totals[label]
            im2col_value = im2col_totals[label]
            kind = "aux" if label == "prepack" else "other" if label == "others" else "site"
            rows.append(
                {
                    "model": model_label,
                    "label": label,
                    "direct_mcycles": f"{direct_value:.3f}",
                    "im2col_mcycles": f"{im2col_value:.3f}",
                    "delta_mcycles": f"{im2col_value - direct_value:.3f}",
                    "kind": kind,
                }
            )

        direct_total = direct.avg_cycles / 1e6
        im2col_total = im2col.avg_cycles / 1e6
        rows.append(
            {
                "model": model_label,
                "label": "total",
                "direct_mcycles": f"{direct_total:.3f}",
                "im2col_mcycles": f"{im2col_total:.3f}",
                "delta_mcycles": f"{im2col_total - direct_total:.3f}",
                "kind": "total",
            }
        )

    path = CSV_DIR / "im2col_site_attribution.csv"
    write_rows(path, columns, rows)
    return path


def write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log(f"wrote {path} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(os.environ.get("PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR", DEFAULT_RESULTS_DIR)),
        help="Result directory produced by scripts/run-firesim-workloads.sh.",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.exists():
        raise SystemExit(f"results directory not found: {results_dir}")

    reset_generated_inputs()
    runs = discover_runs(results_dir)
    if not runs:
        raise SystemExit(f"no model.log files found under {results_dir}")
    log(f"discovered {len(runs)} workload result(s) under {results_dir}")

    prepare_compat_logs(runs)
    generated = [
        write_cnn_result_csv(runs),
        write_sdpa_csv(runs),
        write_flex_prefill_csv(runs),
        write_flash_window_ratio_csv(runs),
        write_im2col_site_csv(runs),
    ]
    log("generated CSV files:")
    for path in generated:
        log(f"  {path}")


if __name__ == "__main__":
    main()
