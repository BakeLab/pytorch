#!/usr/bin/env python3
"""Repair a PyTorch wheel: bundle runtime deps, set RPATHs, retag platform.

Uses auditwheel for unpack/repack (not the `wheel` CLI). `wheel pack`/`wheel tags`
emit an invalid ZIP64 header for archives over 4GB (pypa/wheel#692), which makes
large ROCm wheels fail to install under strict zip parsers such as uv
(pytorch#189748). auditwheel repacks via the stdlib `zipfile`, which writes a
correct ZIP64 record.

Usage: repair_wheel.py <input_dir> <output_dir>

Environment variables:
    DESIRED_CUDA       - cpu, cu126, cu130, xpu, rocm6.4.1, etc.
    GPU_ARCH_TYPE      - cpu, cuda, rocm, xpu
    GPU_ARCH_VERSION   - 12.6, 13.0, 13.2, 6.4.1, etc. (empty for CPU)
    USE_CUDA           - "0" or "1"
    ROCM_HOME          - /opt/rocm (ROCm only)
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from auditwheel.wheeltools import add_platforms, InWheelCtx
from build_env_setup import PLATFORM_TAGS


PATCHELF = "/usr/local/bin/patchelf"


def wheel_platform_tags(wheel_name: str) -> list[str]:
    """Platform tags encoded in a wheel filename.

    The filename is ``{name}-{version}-{python}-{abi}-{platform}.whl``; the
    platform field is the last ``-``-delimited component and may itself be a
    ``.``-joined set of tags (e.g. ``linux_x86_64`` or
    ``manylinux1_x86_64.manylinux_2_17_x86_64``).
    """
    return wheel_name.removesuffix(".whl").rsplit("-", 1)[-1].split(".")


@dataclass
class BundledLib:
    """A compatibility library to copy into torch/lib."""

    src: Path
    dest_name: str  # final filename in torch/lib/


def detect_libgomp() -> Path:
    """libgomp lives in different places on Ubuntu vs RHEL-family images."""
    arch = platform.machine()
    os_release = Path("/etc/os-release").read_text()
    if "Ubuntu" in os_release:
        return Path(f"/usr/lib/{arch}-linux-gnu/libgomp.so.1")
    return Path("/usr/lib64/libgomp.so.1")


def cuda_rpaths(gpu_arch_version: str) -> str:
    """Build the colon-separated RPATH list for CUDA wheels.

    CUDA 13.x bundles all libs under nvidia/cu13/lib; CUDA 12.x ships them
    in per-component packages.
    """
    base = (
        "$ORIGIN/../../nvidia/cudnn/lib"
        ":$ORIGIN/../../nvidia/nvshmem/lib"
        ":$ORIGIN/../../nvidia/nccl/lib"
        ":$ORIGIN/../../nvidia/cusparselt/lib"
    )
    cuda_major = gpu_arch_version.split(".", 1)[0] if gpu_arch_version else ""
    if cuda_major == "13":
        return base + ":$ORIGIN/../../nvidia/cu13/lib"
    return (
        base
        + ":$ORIGIN/../../nvidia/cublas/lib"
        + ":$ORIGIN/../../nvidia/cuda_cupti/lib"
        + ":$ORIGIN/../../nvidia/cuda_nvrtc/lib"
        + ":$ORIGIN/../../nvidia/cuda_runtime/lib"
        + ":$ORIGIN/../../nvidia/cufft/lib"
        + ":$ORIGIN/../../nvidia/cusolver/lib"
        + ":$ORIGIN/../../nvidia/cusparse/lib"
        + ":$ORIGIN/../../cusparselt/lib"
        + ":$ORIGIN/../../nvidia/nvtx/lib"
        + ":$ORIGIN/../../nvidia/cufile/lib"
    )


def rocm_rpaths(rocm_home: Path | None = None) -> str:
    """RPATH list for the TheRock wheel-based ROCm layout.

    ROCm packages are siblings of torch, so paths are ``$ORIGIN``-relative.
    Include flat and discovered per-target LLVM paths for ``libomp.so`` across
    TheRock package layouts.
    """
    rpaths = [
        "$ORIGIN/../../_rocm_sdk_core/lib",
        "$ORIGIN/../../_rocm_sdk_core/lib/rocm_sysdeps/lib",
        "$ORIGIN/../../_rocm_sdk_libraries/lib",
        "$ORIGIN/../../_rocm_sdk_core/lib/llvm/lib",
    ]
    if rocm_home is None:
        env_home = os.environ.get("ROCM_HOME")
        if env_home:
            rocm_home = Path(env_home)
    if rocm_home is not None:
        llvm_lib = rocm_home / "lib" / "llvm" / "lib"
        for libomp in sorted(llvm_lib.glob("*/libomp.so")):
            if libomp.is_file():
                rpaths.append(
                    f"$ORIGIN/../../_rocm_sdk_core/lib/llvm/lib/{libomp.parent.name}"
                )
    return ":".join(rpaths)


def arch_extra_deps(use_cuda: bool) -> list[Path]:
    """CPU builds may link against OpenBLAS; libgfortran comes along either way."""
    candidates: list[Path] = [Path("/usr/lib64/libgfortran.so.5")]
    if not use_cuda:
        candidates.append(Path("/opt/OpenBLAS/lib/libopenblas.so.0"))
    return [p for p in candidates if p.is_file()]


# ROCm shared libs to bundle. Discovered under $ROCM_HOME/{lib,lib64}.
ROCM_SO_FILES: list[str] = [
    "libMIOpen.so",
    "libamdhip64.so",
    "libhipblas.so",
    "libhipfft.so",
    "libhiprand.so",
    "libhipsolver.so",
    "libhipsparse.so",
    "libhsa-runtime64.so",
    "libamd_comgr.so",
    "librccl.so",
    "librocblas.so",
    "librocfft.so",
    "libamd_smi.so",
    "librocrand.so",
    "librocsolver.so",
    "librocsparse.so",
    "libroctracer64.so",
    "libroctx64.so",
    "libhipblaslt.so",
    "libhipsparselt.so",
    "libhiprtc.so",
    "librocprofiler-sdk.so",
    "librocprofiler-register.so",
    "libhsa-amd-aqlprofile64.so",
    "librocm-core.so",
    "librocroller.so",
]

# hipFile only ships with ROCm 7.14 and later, where it is required.
_version_file = Path(os.environ.get("ROCM_HOME", "/opt/rocm")) / ".info" / "version"
_rocm_version = _version_file.read_text().strip() if _version_file.is_file() else ""
if tuple(int(x) for x in _rocm_version.split(".")[:2] if x.isdigit()) >= (7, 14):
    ROCM_SO_FILES.append("libhipfile.so")


def rocm_os_deps() -> list[Path]:
    """OS-side runtime deps that must travel with ROCm wheels."""
    os_release = Path("/etc/os-release").read_text()
    if "Ubuntu" in os_release:
        prefix = "/usr/lib/x86_64-linux-gnu"
        libtinfo = "/lib/x86_64-linux-gnu/libtinfo.so.6"
        libdrm_dir = "/usr/lib/x86_64-linux-gnu"
    else:  # AlmaLinux / CentOS / RHEL
        prefix = "/usr/lib64"
        # CentOS Linux had libtinfo.so.5; AlmaLinux ships .6.
        libtinfo_5 = Path("/usr/lib64/libtinfo.so.5")
        libtinfo = (
            str(libtinfo_5) if libtinfo_5.exists() else "/usr/lib64/libtinfo.so.6"
        )
        libdrm_dir = "/opt/amdgpu/lib64"
    return [
        Path(f"{prefix}/libnuma.so.1"),
        Path(f"{prefix}/libelf.so.1"),
        Path(libtinfo),
        Path(f"{prefix}/libdw.so.1"),
        Path(f"{libdrm_dir}/libdrm.so.2"),
        Path(f"{libdrm_dir}/libdrm_amdgpu.so.1"),
    ]


def find_rocm_lib(rocm_home: Path, basename: str) -> Path | None:
    """Locate a ROCm library, falling back from lib/ to lib64/ to a wider search."""
    for sub in ("lib", "lib64"):
        for hit in (rocm_home / sub).rglob(basename + "*"):
            if hit.is_file() and hit.name.startswith(basename):
                return hit
    for hit in rocm_home.rglob(basename + "*"):
        if hit.is_file() and hit.name.startswith(basename):
            return hit
    return None


def rocm_arch_filter(arch_list: str) -> list[str]:
    return [a for a in arch_list.split(";") if a]


def rocm_lib_kernels(
    rocm_home: Path, lib_subdir: str, archs: list[str]
) -> list[AuxFile]:
    """Per-gfx kernel files under $ROCM_HOME/lib/<lib_subdir>/library/ plus the non-gfx common files."""
    src_dir = rocm_home / "lib" / lib_subdir / "library"
    if not src_dir.is_dir():
        return []
    files: list[AuxFile] = []
    for entry in sorted(src_dir.iterdir()):
        if not entry.is_file():
            continue
        # Pick gfx-specific files matching the arch set, plus common (non-gfx) files.
        name = entry.name
        if "gfx" in name and not any(a in name for a in archs):
            continue
        files.append(AuxFile(src=entry, rel_dest=f"lib/{lib_subdir}/library/{name}"))
    return files


def rocm_bundle(
    rocm_home: Path, gpu_arch_version: str
) -> tuple[list[BundledLib], list[AuxFile]]:
    """Build the ROCm bundle spec: shared libs and auxiliary kernel/db files.

    Versioned ROCm sonames (libfoo.so.6) get renamed to bare .so to match the
    NEEDED entries that hipcc emits, mirroring the original build_rocm.sh
    fname_without_so_number behaviour. NEEDED entries inside the wheel are
    rewritten to the renamed copies via patchelf in repair_wheel().
    """
    libs: list[BundledLib] = []
    so_files = list(ROCM_SO_FILES)
    # librocm_smi64.so is only needed for ROCm7.2 and earlier
    if gpu_arch_version and tuple(map(int, gpu_arch_version.split(".")[:2])) <= (7, 2):
        so_files.append("librocm_smi64.so")
    for stem in so_files:
        path = find_rocm_lib(rocm_home, stem)
        if path is None:
            sys.exit(f"Required ROCm library not found: {stem}")
        # Strip the SO version: libfoo.so.6.1 -> libfoo.so. The ROCm-built
        # binaries in the wheel link against the bare .so SONAME.
        libs.append(BundledLib(src=path, dest_name=stem, needed_alias=stem))
    for os_lib in rocm_os_deps():
        if os_lib.is_file():
            libs.append(BundledLib(src=os_lib, dest_name=os_lib.name))
        else:
            # Silently omitting one of these produces a wheel that imports but
            # misbehaves at runtime, and it took a release candidate to notice.
            print(
                f"WARNING: OS dependency not present on the builder, "
                f"not bundled into the wheel: {os_lib}",
                file=sys.stderr,
            )

    archs = rocm_arch_filter(os.environ.get("PYTORCH_ROCM_ARCH", ""))
    aux: list[AuxFile] = []
    for sub in ("rocblas", "hipblaslt", "hipsparselt"):
        aux += rocm_lib_kernels(rocm_home, sub, archs)
    miopen_db = rocm_home / "share/miopen/db"
    if miopen_db.is_dir():
        for entry in sorted(miopen_db.iterdir()):
            if entry.is_file() and any(a in entry.name for a in archs):
                aux.append(AuxFile(src=entry, rel_dest=f"share/miopen/db/{entry.name}"))
    rccl_dir = rocm_home / "share/rccl/msccl-algorithms"
    if rccl_dir.is_dir():
        for entry in sorted(rccl_dir.iterdir()):
            if entry.is_file():
                aux.append(
                    AuxFile(
                        src=entry,
                        rel_dest=f"share/rccl/msccl-algorithms/{entry.name}",
                    )
                )
    amdgpu_ids = Path("/opt/amdgpu/share/libdrm/amdgpu.ids")
    if amdgpu_ids.is_file():
        aux.append(AuxFile(src=amdgpu_ids, rel_dest="share/libdrm/amdgpu.ids"))
    return libs, aux


def patchelf(*args: str) -> None:
    subprocess.run([PATCHELF, *args], check=True)


def set_rpath(sofile: Path, rpath: str, force_rpath: bool) -> None:
    cmd = ["--set-rpath", rpath]
    if force_rpath:
        cmd.append("--force-rpath")
    cmd.append(str(sofile))
    patchelf(*cmd)


def repair_wheel(
    wheel: Path,
    output_dir: Path,
    platform_tag: str,
    libgomp_path: Path,
    arch_deps: list[Path],
    bundled_libs: list[BundledLib],
    c_so_rpath: str,
    lib_so_rpath: str,
    force_rpath: bool,
) -> None:
    # InWheelCtx unpacks via auditwheel's zip2dir on enter and, once out_wheel is
    # set, regenerates RECORD and repacks via dir2zip on exit. dir2zip uses the
    # stdlib zipfile (correct ZIP64), unlike `wheel pack`/`wheel tags` which
    # corrupt >4GB ROCm wheels (pypa/wheel#692, pytorch#189748).
    with InWheelCtx(wheel, output_dir / wheel.name) as ctx:
        torch_dir = ctx.path / "torch"
        torch_lib = torch_dir / "lib"

        # Bundle libgomp and rewrite NEEDED entries to point at our copy
        shutil.copy(libgomp_path, torch_lib / "libgomp.so.1")
        for sofile in torch_dir.glob("*.so*"):
            if sofile.is_file():
                patchelf(
                    "--replace-needed",
                    "libgomp.so.1",
                    "libgomp.so.1",
                    str(sofile),
                )

        # Bundle extra BLAS/LAPACK dependencies (usually none)
        for dep in arch_deps:
            shutil.copy(dep, torch_lib / dep.name)

        # Keep only compatibility dependencies that are not supplied by the
        # ROCm SDK wheel packages (currently pre-10.0 rocSHMEM's libnuma).
        for lib in bundled_libs:
            shutil.copy(lib.src, torch_lib / lib.dest_name)
        # Set RPATH on top-level (_C.so etc.) and lib/ shared objects
        for sofile in torch_dir.glob("*.so*"):
            if sofile.is_file():
                set_rpath(sofile, c_so_rpath, force_rpath)
        for sofile in torch_lib.glob("*.so*"):
            if sofile.is_file():
                set_rpath(sofile, lib_so_rpath, force_rpath)

        # Retag linux_* -> manylinux_2_28_* in both the filename and the WHEEL
        # metadata. add_platforms updates ctx.out_wheel to the new name; RECORD
        # regeneration and repacking happen when the context exits.
        add_platforms(
            ctx, [platform_tag], remove_platforms=wheel_platform_tags(wheel.name)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    arch = platform.machine()
    use_cuda = os.environ.get("USE_CUDA", "0") == "1"
    gpu_arch_type = os.environ.get("GPU_ARCH_TYPE", "")
    gpu_arch_version = os.environ.get("GPU_ARCH_VERSION", "")
    is_rocm = gpu_arch_type == "rocm" or "rocm" in os.environ.get("DESIRED_CUDA", "")

    libgomp_path = detect_libgomp()
    if not libgomp_path.exists():
        sys.exit(f"libgomp not found at {libgomp_path}")

    bundled_libs: list[BundledLib] = []

    if use_cuda:
        rpaths = cuda_rpaths(gpu_arch_version)
        c_so_rpath = f"{rpaths}:$ORIGIN:$ORIGIN/lib"
        lib_so_rpath = f"{rpaths}:$ORIGIN"
        force_rpath = True
    elif gpu_arch_type == "xpu":
        # XPU runtime libs come from pypi packages; set RPATHs like CUDA.
        xpu_rpaths = "$ORIGIN/../../../.."
        c_so_rpath = f"{xpu_rpaths}:$ORIGIN:$ORIGIN/lib"
        lib_so_rpath = f"{xpu_rpaths}:$ORIGIN"
        force_rpath = True
    elif is_rocm:
        rocm_home = Path(os.environ.get("ROCM_HOME", "/opt/rocm"))
        # A classic install has no _rocm_sdk_* siblings, so SDK RPATHs
        # would not resolve. CI images always use the TheRock layout.
        if "_rocm_sdk" not in str(rocm_home):
            sys.exit(
                f"Only the TheRock SDK layout is supported (ROCM_HOME={rocm_home})"
            )
        rpaths = rocm_rpaths(rocm_home)
        c_so_rpath = f"{rpaths}:$ORIGIN:$ORIGIN/lib"
        lib_so_rpath = f"{rpaths}:$ORIGIN"
        force_rpath = True
        # Pre-10.0 TheRock SDKs ship a rocSHMEM that dlopens the bare name
        # "libnuma.so". At wheel runtime that name resolves nowhere: the
        # SDK vendors numa only as librocm_sysdeps_numa.so.1, and on user
        # systems the bare dev name exists only if numactl-devel is
        # installed (#195670). The 10.0 SDK line links the vendored soname
        # directly instead (rocm-systems#6640), so bundle a bare-named
        # copy, reached via $ORIGIN on the RPATHs above, only for older
        # SDKs. The builder image installs numactl-libs for this.
        ver = gpu_arch_version
        if not ver or tuple(map(int, ver.split(".")[:2])) < (10, 0):
            is_ubuntu = "Ubuntu" in Path("/etc/os-release").read_text()
            libdir = "/usr/lib/x86_64-linux-gnu" if is_ubuntu else "/usr/lib64"
            libnuma = Path(libdir) / "libnuma.so.1"
            if not libnuma.is_file():
                sys.exit(f"libnuma to bundle for rocSHMEM not found: {libnuma}")
            bundled_libs.append(BundledLib(src=libnuma, dest_name="libnuma.so"))
    else:
        c_so_rpath = "$ORIGIN:$ORIGIN/lib"
        lib_so_rpath = "$ORIGIN"
        force_rpath = False

    arch_deps = arch_extra_deps(use_cuda)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    wheels = sorted(args.input_dir.glob("*.whl"))
    if not wheels:
        sys.exit(f"No wheels found in {args.input_dir}")

    if arch not in PLATFORM_TAGS:
        sys.exit(f"Unknown arch {arch}")
    platform_tag = PLATFORM_TAGS[arch]
    for whl in wheels:
        repair_wheel(
            whl,
            args.output_dir,
            platform_tag,
            libgomp_path,
            arch_deps,
            bundled_libs,
            c_so_rpath,
            lib_so_rpath,
            force_rpath,
        )

    repaired = list(args.output_dir.glob("*.whl"))
    print(f"Repaired {len(repaired)} wheel(s) in {args.output_dir}")


if __name__ == "__main__":
    main()
