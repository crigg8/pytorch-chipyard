#!/usr/bin/env python3
"""Record and summarize Table 2 compilation/simulation measurements."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path


RAW_FIELDS = [
    "run_id",
    "timestamp_utc",
    "trial",
    "workload",
    "toolchain",
    "phase",
    "simulator",
    "total_wall_s",
    "kernel_count",
    "per_kernel_wall_s",
    "status",
    "exit_code",
    "artifact_path",
    "log_path",
    "notes",
]

SUMMARY_FIELDS = [
    "workload",
    "toolchain",
    "kernel_count",
    "compile_total_wall_s",
    "compile_per_kernel_s",
    "spike_wall_s",
    "firesim_wall_s",
    "compile_trials",
    "spike_trials",
    "firesim_trials",
    "status",
]

WORKLOAD_ORDER = {
    "SqueezeNet": 0,
    "AlexNet": 1,
    "MobileNetV2": 2,
    "ResNet50": 3,
}
TOOLCHAIN_ORDER = {"PyTorch-Chipyard": 0, "TVM-Gemmini": 1}


def _format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def _float_or_none(value: str) -> float | None:
    if not value:
        return None
    return float(value)


def _int_or_none(value: str) -> int | None:
    if not value:
        return None
    return int(value)


def init_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        with path.open(newline="") as csv_file:
            header = next(csv.reader(csv_file), [])
        if header != RAW_FIELDS:
            raise SystemExit(f"unexpected CSV header in {path}: {header}")
        return
    with path.open("w", newline="") as csv_file:
        csv.DictWriter(csv_file, fieldnames=RAW_FIELDS).writeheader()


def append_row(args: argparse.Namespace) -> None:
    path = Path(args.csv)
    init_csv(path)
    kernel_count = _int_or_none(args.kernel_count)
    total_wall_s = _float_or_none(args.total_wall_s)
    per_kernel_wall_s = None
    if args.phase == "compile" and total_wall_s is not None and kernel_count:
        per_kernel_wall_s = total_wall_s / kernel_count

    row = {
        "run_id": args.run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "trial": args.trial,
        "workload": args.workload,
        "toolchain": args.toolchain,
        "phase": args.phase,
        "simulator": args.simulator,
        "total_wall_s": _format_float(total_wall_s),
        "kernel_count": "" if kernel_count is None else str(kernel_count),
        "per_kernel_wall_s": _format_float(per_kernel_wall_s),
        "status": args.status,
        "exit_code": args.exit_code,
        "artifact_path": args.artifact_path,
        "log_path": args.log_path,
        "notes": args.notes,
    }
    with path.open("a", newline="") as csv_file:
        csv.DictWriter(csv_file, fieldnames=RAW_FIELDS).writerow(row)


def _median(rows: list[dict[str, str]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field)]
    return statistics.median(values) if values else None


def summarize(raw_path: Path, output_path: Path) -> None:
    with raw_path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    keys = sorted(
        {(row["workload"], row["toolchain"]) for row in rows},
        key=lambda key: (
            WORKLOAD_ORDER.get(key[0], len(WORKLOAD_ORDER)),
            TOOLCHAIN_ORDER.get(key[1], len(TOOLCHAIN_ORDER)),
            key,
        ),
    )
    summaries: list[dict[str, str]] = []
    for workload, toolchain in keys:
        matching_history = [
            row
            for row in rows
            if row["workload"] == workload and row["toolchain"] == toolchain
        ]
        # A resumed run appends a new row for a previously failed measurement.
        # Only the latest attempt for each trial/phase/simulator is active;
        # otherwise an old FAIL would poison the summary after a successful
        # retry forever.
        latest: dict[tuple[str, str, str], dict[str, str]] = {}
        for row in matching_history:
            latest[(row["trial"], row["phase"], row["simulator"])] = row
        matching = list(latest.values())
        passed = [row for row in matching if row["status"] == "PASS"]
        compile_rows = [row for row in passed if row["phase"] == "compile"]
        spike_rows = [row for row in passed if row["simulator"] == "spike"]
        firesim_rows = [row for row in passed if row["simulator"] == "firesim"]
        kernel_counts = {
            int(row["kernel_count"]) for row in compile_rows if row["kernel_count"]
        }
        kernel_count = ""
        if len(kernel_counts) == 1:
            kernel_count = str(next(iter(kernel_counts)))
        elif len(kernel_counts) > 1:
            kernel_count = "mixed"

        status = "PASS"
        if any(row["status"] == "FAIL" for row in matching):
            status = "FAIL"
        elif not passed:
            status = "MISSING"

        summaries.append(
            {
                "workload": workload,
                "toolchain": toolchain,
                "kernel_count": kernel_count,
                "compile_total_wall_s": _format_float(
                    _median(compile_rows, "total_wall_s")
                ),
                "compile_per_kernel_s": _format_float(
                    _median(compile_rows, "per_kernel_wall_s")
                ),
                "spike_wall_s": _format_float(
                    _median(spike_rows, "total_wall_s")
                ),
                "firesim_wall_s": _format_float(
                    _median(firesim_rows, "total_wall_s")
                ),
                "compile_trials": str(len(compile_rows)),
                "spike_trials": str(len(spike_rows)),
                "firesim_trials": str(len(firesim_rows)),
                "status": status,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)


def count_pytorch_kernels(path: Path) -> int:
    with path.open() as model_spec_file:
        model_spec = json.load(model_spec_file)
    kernels = model_spec.get("kernels")
    if not isinstance(kernels, list) or not kernels:
        raise SystemExit(f"no kernels found in {path}")
    return len(kernels)


TVM_FUNCTION = re.compile(
    r"^TVM_DLL int32_t (tvmgen_default_[A-Za-z0-9_]+)\([^\n]*\) \{$",
    re.MULTILINE,
)


def count_tvm_kernels(model_dir: Path) -> int:
    functions: set[str] = set()
    for path in model_dir.glob("default_lib*.c"):
        functions.update(TVM_FUNCTION.findall(path.read_text(errors="replace")))
    functions.discard("tvmgen_default___tvm_main__")
    if not functions:
        raise SystemExit(f"no TVM compute functions found under {model_dir}")
    return len(functions)


def has_passed_measurement(
    path: Path,
    *,
    trial: str,
    workload: str,
    toolchain: str,
    phase: str,
    simulator: str,
) -> bool:
    if not path.is_file():
        return False
    with path.open(newline="") as csv_file:
        rows = csv.DictReader(csv_file)
        return any(
            row["trial"] == trial
            and row["workload"] == workload
            and row["toolchain"] == toolchain
            and row["phase"] == phase
            and row["simulator"] == simulator
            and row["status"] == "PASS"
            for row in rows
        )


def extract_last(path: Path, key: str) -> str:
    pattern = re.compile(
        rf"(?:^|\[compile\]\s+){re.escape(key)}=([0-9]+(?:\.[0-9]+)?)",
        re.MULTILINE,
    )
    matches = pattern.findall(path.read_text(errors="replace"))
    if not matches:
        raise SystemExit(f"{key}=... not found in {path}")
    return matches[-1]


def extract_firesim_wall(path: Path) -> str:
    """Extract a successful FireSim host-wall measurement from a saved log."""
    text = path.read_text(errors="replace")

    # run_table2.sh writes this marker only after a successful fallback run.
    marker_matches = re.findall(
        r"^TABLE2_FIRESIM_WALL_S=([0-9]+(?:\.[0-9]+)?)$", text, re.MULTILINE
    )
    if marker_matches:
        return marker_matches[-1]

    # A collected FireSim uartlog contains the simulator's authoritative wall
    # time.  Do not reuse an incomplete/failed UART just because it printed a
    # partial performance summary.
    if "*** PASSED ***" not in text:
        raise SystemExit(f"FireSim PASS marker not found in {path}")
    summary_matches = re.findall(
        r"^Wallclock Time Elapsed:\s*([0-9]+(?:\.[0-9]+)?)\s*s\s*$",
        text,
        re.MULTILINE,
    )
    if not summary_matches:
        raise SystemExit(f"FireSim wallclock summary not found in {path}")
    return summary_matches[-1]


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--csv", required=True)

    append_parser = subparsers.add_parser("append")
    append_parser.add_argument("--csv", required=True)
    append_parser.add_argument("--run-id", required=True)
    append_parser.add_argument("--trial", required=True)
    append_parser.add_argument("--workload", required=True)
    append_parser.add_argument("--toolchain", required=True)
    append_parser.add_argument("--phase", choices=["compile", "simulation"], required=True)
    append_parser.add_argument("--simulator", choices=["", "spike", "firesim"], default="")
    append_parser.add_argument("--total-wall-s", default="")
    append_parser.add_argument("--kernel-count", default="")
    append_parser.add_argument("--status", choices=["PASS", "FAIL", "SKIP"], required=True)
    append_parser.add_argument("--exit-code", default="")
    append_parser.add_argument("--artifact-path", default="")
    append_parser.add_argument("--log-path", default="")
    append_parser.add_argument("--notes", default="")

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--csv", required=True)
    summary_parser.add_argument("--output", required=True)

    count_pc_parser = subparsers.add_parser("count-pytorch")
    count_pc_parser.add_argument("path")

    count_tvm_parser = subparsers.add_parser("count-tvm")
    count_tvm_parser.add_argument("path")

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--log", required=True)
    extract_parser.add_argument("--key", required=True)

    firesim_parser = subparsers.add_parser("extract-firesim-wall")
    firesim_parser.add_argument("--log", required=True)

    has_pass_parser = subparsers.add_parser("has-pass")
    has_pass_parser.add_argument("--csv", required=True)
    has_pass_parser.add_argument("--trial", required=True)
    has_pass_parser.add_argument("--workload", required=True)
    has_pass_parser.add_argument("--toolchain", required=True)
    has_pass_parser.add_argument(
        "--phase", choices=["compile", "simulation"], required=True
    )
    has_pass_parser.add_argument(
        "--simulator", choices=["", "spike", "firesim"], default=""
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "init":
        init_csv(Path(args.csv))
    elif args.command == "append":
        append_row(args)
    elif args.command == "summarize":
        summarize(Path(args.csv), Path(args.output))
    elif args.command == "count-pytorch":
        print(count_pytorch_kernels(Path(args.path)))
    elif args.command == "count-tvm":
        print(count_tvm_kernels(Path(args.path)))
    elif args.command == "extract":
        print(extract_last(Path(args.log), args.key))
    elif args.command == "extract-firesim-wall":
        print(extract_firesim_wall(Path(args.log)))
    elif args.command == "has-pass":
        raise SystemExit(
            0
            if has_passed_measurement(
                Path(args.csv),
                trial=args.trial,
                workload=args.workload,
                toolchain=args.toolchain,
                phase=args.phase,
                simulator=args.simulator,
            )
            else 1
        )


if __name__ == "__main__":
    main()
