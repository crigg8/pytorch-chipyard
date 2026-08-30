from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(4.0, 3.0), facecolor="white")
    figure.savefig(output, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
