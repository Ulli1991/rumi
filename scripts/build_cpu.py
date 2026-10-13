"""Build the standalone CPU regularization library without Kokkos or Python headers."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output or root / "build" / ("librumi_cpu.dylib" if sys.platform == "darwin" else "librumi_cpu.so")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = shlex.split(os.environ.get("CXX", "c++")) + ["-std=c++17", "-O3", "-fPIC", "-shared",
                  str(root / "src/rumi/native/logh.cpp"), "-o", str(output)]
    subprocess.run(command, check=True)
    print(output)


if __name__ == "__main__":
    main()
