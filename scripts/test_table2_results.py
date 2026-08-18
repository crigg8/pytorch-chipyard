#!/usr/bin/env python3

import csv
import json
import tempfile
import unittest
from pathlib import Path

import table2_results


class TestTable2Results(unittest.TestCase):
    def test_extract_firesim_wall_from_uart_or_table2_marker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            uart = root / "uartlog"
            uart.write_text(
                "Simulation complete.\n"
                "*** PASSED *** after 123 cycles\n"
                "Wallclock Time Elapsed: 981.2 s\n"
            )
            self.assertEqual(table2_results.extract_firesim_wall(uart), "981.2")

            cached = root / "table2-firesim.log"
            cached.write_text("TABLE2_FIRESIM_WALL_S=42.125000\n")
            self.assertEqual(
                table2_results.extract_firesim_wall(cached), "42.125000"
            )

            failed = root / "failed-uartlog"
            failed.write_text("Wallclock Time Elapsed: 1.0 s\n")
            with self.assertRaises(SystemExit):
                table2_results.extract_firesim_wall(failed)

    def test_kernel_counters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_spec = root / "model_spec.json"
            model_spec.write_text(json.dumps({"kernels": [{}, {}, {}]}))
            self.assertEqual(table2_results.count_pytorch_kernels(model_spec), 3)

            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "default_lib1.c").write_text(
                "TVM_DLL int32_t tvmgen_default_fused_a(void* x) {\n}\n"
                "TVM_DLL int32_t tvmgen_default_fused_b(void* x) {\n}\n"
                "TVM_DLL int32_t tvmgen_default___tvm_main__(void* x) {\n}\n"
            )
            self.assertEqual(table2_results.count_tvm_kernels(model_dir), 2)

    def test_summary_uses_successful_trial_medians(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_path = root / "raw.csv"
            summary_path = root / "table2.csv"
            table2_results.init_csv(raw_path)

            base = {
                "csv": str(raw_path),
                "run_id": "test",
                "workload": "AlexNet",
                "toolchain": "PyTorch-Chipyard",
                "exit_code": "0",
                "artifact_path": "artifact",
                "log_path": "log",
                "notes": "",
            }
            rows = [
                ("1", "compile", "", "9", "3"),
                ("2", "compile", "", "15", "3"),
                ("1", "simulation", "spike", "5", ""),
                ("2", "simulation", "spike", "7", ""),
                ("1", "simulation", "firesim", "20", ""),
            ]
            for trial, phase, simulator, wall_s, kernel_count in rows:
                args = type(
                    "Args",
                    (),
                    {
                        **base,
                        "trial": trial,
                        "phase": phase,
                        "simulator": simulator,
                        "total_wall_s": wall_s,
                        "kernel_count": kernel_count,
                        "status": "PASS",
                    },
                )()
                table2_results.append_row(args)

            table2_results.summarize(raw_path, summary_path)
            with summary_path.open(newline="") as csv_file:
                row = next(csv.DictReader(csv_file))
            self.assertEqual(row["kernel_count"], "3")
            self.assertEqual(row["compile_total_wall_s"], "12.000000")
            self.assertEqual(row["compile_per_kernel_s"], "4.000000")
            self.assertEqual(row["spike_wall_s"], "6.000000")
            self.assertEqual(row["firesim_wall_s"], "20.000000")


if __name__ == "__main__":
    unittest.main()
