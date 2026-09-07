#!/usr/bin/env python3

import argparse
import os
import platform
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from urllib.parse import urlsplit


ACCELERATORS = ("cuda", "rocm", "xpu", "mps")
DEFAULT_BASE_URL = "https://bb-pypi.int.bb.pub/bakeai/dev"
ACCELERATOR_OVERRIDE = "PYTORCH_ACCELERATOR"


def _validate_accelerator(accelerator: str) -> str:
    if accelerator not in ACCELERATORS:
        choices = ", ".join(ACCELERATORS)
        raise ValueError(f"unsupported accelerator {accelerator!r}; expected one of {choices}")
    return accelerator


def _has_intel_drm_device(sys_root: Path) -> bool:
    drm_root = sys_root / "class" / "drm"
    for vendor_file in drm_root.glob("*/device/vendor"):
        try:
            if vendor_file.read_text().strip().lower() == "0x8086":
                return True
        except OSError:
            continue
    return False


def detect_accelerator(
    *,
    environ: Mapping[str, str] = os.environ,
    dev_root: Path = Path("/dev"),
    proc_root: Path = Path("/proc"),
    sys_root: Path = Path("/sys"),
    system: str | None = None,
    machine: str | None = None,
) -> str:
    override = environ.get(ACCELERATOR_OVERRIDE)
    if override:
        return _validate_accelerator(override.lower())

    system = system or platform.system()
    machine = machine or platform.machine()
    if system == "Darwin" and machine == "arm64":
        return "mps"
    if system != "Linux":
        raise RuntimeError(f"no supported accelerator package for {system} {machine}")

    has_rocm = (dev_root / "kfd").exists()
    has_cuda = (dev_root / "nvidiactl").exists() or (
        proc_root / "driver" / "nvidia" / "version"
    ).exists()
    if has_rocm and has_cuda:
        raise RuntimeError(
            f"both ROCm and CUDA devices were detected; set {ACCELERATOR_OVERRIDE}"
        )
    if has_rocm:
        return "rocm"
    if has_cuda:
        return "cuda"
    if _has_intel_drm_device(sys_root):
        return "xpu"
    raise RuntimeError(
        f"no supported accelerator was detected; set {ACCELERATOR_OVERRIDE}"
    )


def _normalize_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("the accelerator index base URL must be an HTTPS URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "the accelerator index base URL must not contain credentials, a query, or a fragment"
        )
    return normalized


def repository_url(base_url: str, accelerator: str) -> str:
    return f"{_normalize_base_url(base_url)}/{_validate_accelerator(accelerator)}"


def simple_index_url(base_url: str, accelerator: str) -> str:
    return f"{repository_url(base_url, accelerator)}/simple"


def wheel_accelerator(wheel: Path) -> str:
    parts = wheel.name.removesuffix(".whl").split("-")
    if wheel.suffix != ".whl" or len(parts) < 5 or parts[0] != "torch":
        raise ValueError(f"expected a torch wheel, got {wheel.name!r}")
    _, separator, local_version = parts[1].partition("+")
    if not separator:
        raise ValueError(
            f"torch wheel {wheel.name!r} does not encode an accelerator local version"
        )
    accelerator = local_version.split(".", 1)[0].lower()
    return _validate_accelerator(accelerator)


def _resolve_accelerator(requested: str) -> str:
    if requested == "auto":
        return detect_accelerator()
    return _validate_accelerator(requested)


def _pip_scope(requested: str) -> str:
    if requested != "auto":
        return requested
    return "site" if sys.prefix != sys.base_prefix else "user"


def configure_pip(
    *,
    base_url: str,
    accelerator: str,
    scope: str,
    dry_run: bool,
) -> None:
    accelerator = _resolve_accelerator(accelerator)
    index_url = simple_index_url(base_url, accelerator)
    command = [
        sys.executable,
        "-m",
        "pip",
        "config",
        f"--{_pip_scope(scope)}",
        "set",
        "global.index-url",
        index_url,
    ]
    print(f"accelerator: {accelerator}")
    print(f"index-url: {index_url}")
    if dry_run:
        print(shlex.join(command))
        return
    subprocess.run(command, check=True)


def publish_wheels(
    wheels: Sequence[Path],
    *,
    base_url: str,
    dry_run: bool,
) -> None:
    if not wheels:
        raise ValueError("at least one wheel is required")
    missing = [str(wheel) for wheel in wheels if not wheel.is_file()]
    if missing:
        raise FileNotFoundError(f"wheel files do not exist: {', '.join(missing)}")

    accelerators = {wheel_accelerator(wheel) for wheel in wheels}
    if len(accelerators) != 1:
        raise ValueError("all wheels in one upload must target the same accelerator")
    accelerator = accelerators.pop()
    upload_url = repository_url(base_url, accelerator)
    command = [
        sys.executable,
        "-m",
        "twine",
        "upload",
        "--repository-url",
        upload_url,
        *(str(wheel) for wheel in wheels),
    ]
    print(f"accelerator: {accelerator}")
    print(f"repository-url: {upload_url}")
    if dry_run:
        print(shlex.join(command))
        return
    subprocess.run(command, check=True)


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Configure or publish to the internal accelerator-specific torch indexes."
    )
    parser.set_defaults(base_url=DEFAULT_BASE_URL)
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect = subparsers.add_parser("detect", help="print the detected accelerator")
    detect.set_defaults(handler=lambda args: print(detect_accelerator()))

    configure = subparsers.add_parser(
        "configure", help="configure pip to install torch from the detected channel"
    )
    configure.add_argument("--base-url", default=DEFAULT_BASE_URL)
    configure.add_argument(
        "--accelerator", choices=("auto", *ACCELERATORS), default="auto"
    )
    configure.add_argument(
        "--scope", choices=("auto", "user", "site", "global"), default="auto"
    )
    configure.add_argument("--dry-run", action="store_true")
    configure.set_defaults(
        handler=lambda args: configure_pip(
            base_url=args.base_url,
            accelerator=args.accelerator,
            scope=args.scope,
            dry_run=args.dry_run,
        )
    )

    publish = subparsers.add_parser(
        "publish", help="upload torch wheels to the channel encoded in their version"
    )
    publish.add_argument("wheels", nargs="+", type=Path)
    publish.add_argument("--base-url", default=DEFAULT_BASE_URL)
    publish.add_argument("--dry-run", action="store_true")
    publish.set_defaults(
        handler=lambda args: publish_wheels(
            args.wheels, base_url=args.base_url, dry_run=args.dry_run
        )
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _create_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
