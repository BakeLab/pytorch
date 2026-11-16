#!/usr/bin/env python3
"""Install the repository's declared build requirements with the selected Python."""

import subprocess
import sys
import tomllib
from pathlib import Path


def main() -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with pyproject.open("rb") as source:
        requirements = tomllib.load(source)["build-system"]["requires"]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", *requirements], check=True
    )


if __name__ == "__main__":
    main()
