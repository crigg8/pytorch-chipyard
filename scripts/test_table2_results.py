#!/usr/bin/env python3

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import table2_results


SPEC = Path(__file__).resolve().parents[1] / "benchmarks" / "table2-kernels.json"
KERNEL_ID = "resnet50_classifier"


class TestTable2Results(unittest.TestCase):
    def append(
        self,
        raw_path: Path,
        *,
        trial: str,
        toolchain: str,
        phase: str,
        simulator: str = "",
        wall: str = "1",
        status: str = "PASS",
        exit_code: str = "0",
    ) -> None:
        table2_results.append_row(
            SimpleNamespace(
                csv=str(raw_path),
                run_id="test",
                trial=trial,
                kernel_id=KERNEL_ID,
                kernel_label="ResNet-50 classifier",
                source_model="ResNet-50",
                shape="1x1000x2048",
                macs="2048000",
                toolchain=toolchain,
                phase=phase,
                simulator=simulator,
                total_wall_s=wall,
                status=status,
                exit_code=exit_code,
                artifact_path="artifact",
                log_path="log",
                target_metadata="target",
                notes="",
            )
        )

    def summary_row(self, raw_path: Path, output_path: Path) -> dict[str, str]:
        table2_results.summarize(raw_path, SPEC, output_path)
        with output_path.open(newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))
        return next(row for row in rows if row["kernel_id"] == KERNEL_ID)

    def test_extract_firesim_wall_requires_pass(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            uart = root / "uartlog"
            uart.write_text(
                "Simulation complete.\n"
                "*** PASSED *** after 123 cycles\n"
                "Wallclock Time Elapsed: 981.2 s\n"
            )
            self.assertEqual(table2_results.extract_firesim_wall(uart), "981.2")

            failed = root / "failed-uartlog"
            failed.write_text("Wallclock Time Elapsed: 1.0 s\n")
            with self.assertRaises(SystemExit):
                table2_results.extract_firesim_wall(failed)

    def test_summary_uses_successful_medians_for_four_measurements(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_path = root / "raw.csv"
            table2_results.init_csv(raw_path)
            measurements = [
                ("PyTorch-Chipyard", "compile", "", "10", "14"),
                ("PyTorch-Chipyard", "rtl", "firesim", "20", "24"),
                ("TVM-Gemmini", "compile", "", "2", "4"),
                ("TVM-Gemmini", "rtl", "verilator", "100", "140"),
            ]
            for toolchain, phase, simulator, first, second in measurements:
                self.append(
                    raw_path,
                    trial="1",
                    toolchain=toolchain,
                    phase=phase,
                    simulator=simulator,
                    wall=first,
                )
                self.append(
                    raw_path,
                    trial="2",
                    toolchain=toolchain,
                    phase=phase,
                    simulator=simulator,
                    wall=second,
                )

            row = self.summary_row(raw_path, root / "table2.csv")
            self.assertEqual(row["pytorch_compile_s"], "12.000000")
            self.assertEqual(row["firesim_s"], "22.000000")
            self.assertEqual(row["tvm_compile_s"], "3.000000")
            self.assertEqual(row["verilator_s"], "120.000000")
            self.assertEqual(row["status"], "PASS")

    def test_successful_retry_supersedes_failed_attempt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_path = root / "raw.csv"
            table2_results.init_csv(raw_path)
            measurements = [
                ("PyTorch-Chipyard", "compile", "", "10"),
                ("PyTorch-Chipyard", "rtl", "firesim", "20"),
                ("TVM-Gemmini", "compile", "", "2"),
                ("TVM-Gemmini", "rtl", "verilator", "100"),
            ]
            for toolchain, phase, simulator, wall in measurements:
                self.append(
                    raw_path,
                    trial="1",
                    toolchain=toolchain,
                    phase=phase,
                    simulator=simulator,
                    wall=wall,
                )
            self.append(
                raw_path,
                trial="1",
                toolchain="TVM-Gemmini",
                phase="rtl",
                simulator="verilator",
                wall="",
                status="FAIL",
                exit_code="19",
            )
            failed = self.summary_row(raw_path, root / "failed.csv")
            self.assertEqual(failed["status"], "FAIL")

            self.append(
                raw_path,
                trial="1",
                toolchain="TVM-Gemmini",
                phase="rtl",
                simulator="verilator",
                wall="120",
            )
            recovered = self.summary_row(raw_path, root / "recovered.csv")
            self.assertEqual(recovered["status"], "PASS")
            self.assertEqual(recovered["verilator_s"], "120.000000")

    def test_has_passed_measurement_matches_exact_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_path = Path(temp_dir) / "raw.csv"
            table2_results.init_csv(raw_path)
            self.append(
                raw_path,
                trial="1",
                toolchain="PyTorch-Chipyard",
                phase="compile",
            )
            self.assertTrue(
                table2_results.has_passed_measurement(
                    raw_path,
                    trial="1",
                    kernel_id=KERNEL_ID,
                    toolchain="PyTorch-Chipyard",
                    phase="compile",
                    simulator="",
                )
            )
            self.assertFalse(
                table2_results.has_passed_measurement(
                    raw_path,
                    trial="1",
                    kernel_id=KERNEL_ID,
                    toolchain="PyTorch-Chipyard",
                    phase="rtl",
                    simulator="firesim",
                )
            )

    def test_latex_uses_seconds_and_na(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_path = root / "raw.csv"
            summary_path = root / "table2.csv"
            output_path = root / "rows.tex"
            table2_results.init_csv(raw_path)
            self.append(
                raw_path,
                trial="1",
                toolchain="PyTorch-Chipyard",
                phase="compile",
                wall="1.25",
            )
            table2_results.summarize(raw_path, SPEC, summary_path)
            table2_results.write_latex(summary_path, output_path)
            text = output_path.read_text()
            self.assertIn(r"1.2\,s", text)
            self.assertIn("N/A", text)


if __name__ == "__main__":
    unittest.main()
