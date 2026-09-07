# Owner(s): ["oncall: releng"]

import sys
import tempfile
from pathlib import Path

from torch.testing._internal.common_utils import run_tests, TestCase


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.packaging.accelerator_index import (
    detect_accelerator,
    repository_url,
    simple_index_url,
    wheel_accelerator,
)

sys.path.remove(str(REPO_ROOT))


class TestAcceleratorIndex(TestCase):
    def test_accelerator_override(self):
        self.assertEqual(
            detect_accelerator(environ={"PYTORCH_ACCELERATOR": "ROCM"}), "rocm"
        )

    def test_detect_mps(self):
        self.assertEqual(
            detect_accelerator(environ={}, system="Darwin", machine="arm64"), "mps"
        )

    def test_detect_linux_accelerators(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dev_root = root / "dev"
            proc_root = root / "proc"
            sys_root = root / "sys"
            dev_root.mkdir()

            (dev_root / "kfd").touch()
            self.assertEqual(
                detect_accelerator(
                    environ={},
                    dev_root=dev_root,
                    proc_root=proc_root,
                    sys_root=sys_root,
                    system="Linux",
                    machine="x86_64",
                ),
                "rocm",
            )

            (dev_root / "kfd").unlink()
            (dev_root / "nvidiactl").touch()
            vendor = sys_root / "class" / "drm" / "card0" / "device" / "vendor"
            vendor.parent.mkdir(parents=True)
            vendor.write_text("0x8086\n")
            self.assertEqual(
                detect_accelerator(
                    environ={},
                    dev_root=dev_root,
                    proc_root=proc_root,
                    sys_root=sys_root,
                    system="Linux",
                    machine="x86_64",
                ),
                "cuda",
            )

            (dev_root / "nvidiactl").unlink()
            self.assertEqual(
                detect_accelerator(
                    environ={},
                    dev_root=dev_root,
                    proc_root=proc_root,
                    sys_root=sys_root,
                    system="Linux",
                    machine="x86_64",
                ),
                "xpu",
            )

    def test_reject_ambiguous_linux_accelerators(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dev_root = root / "dev"
            dev_root.mkdir()
            (dev_root / "kfd").touch()
            (dev_root / "nvidiactl").touch()
            with self.assertRaisesRegex(RuntimeError, "both ROCm and CUDA"):
                detect_accelerator(
                    environ={},
                    dev_root=dev_root,
                    proc_root=root / "proc",
                    sys_root=root / "sys",
                    system="Linux",
                    machine="x86_64",
                )

    def test_channel_urls(self):
        base_url = "https://packages.example.test/team/dev/"
        self.assertEqual(
            repository_url(base_url, "rocm"),
            "https://packages.example.test/team/dev/rocm",
        )
        self.assertEqual(
            simple_index_url(base_url, "rocm"),
            "https://packages.example.test/team/dev/rocm/simple",
        )

    def test_wheel_accelerator(self):
        wheels = {
            "torch-2.15.0+cuda.13.4.g0123abcd-cp314-cp314-linux_x86_64.whl": "cuda",
            "torch-2.15.0+rocm.10.0.gfx1201.g0123abcd-cp314-cp314-linux_x86_64.whl": "rocm",
            "torch-2.15.0+xpu.bmg.g0123abcd-cp314-cp314-linux_x86_64.whl": "xpu",
            "torch-2.15.0+mps.apple.silicon.g0123abcd-cp314-cp314-macosx_15_0_arm64.whl": "mps",
        }
        for filename, expected in wheels.items():
            with self.subTest(filename=filename):
                self.assertEqual(wheel_accelerator(Path(filename)), expected)

    def test_reject_wheel_without_accelerator(self):
        with self.assertRaisesRegex(ValueError, "does not encode"):
            wheel_accelerator(
                Path("torch-2.15.0-cp314-cp314-linux_x86_64.whl")
            )


if __name__ == "__main__":
    run_tests()
