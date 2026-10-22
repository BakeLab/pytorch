# Owner(s): ["module: cuda"]

import os
import tempfile

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    run_tests,
    TemporaryFileName,
    TestCase,
)


class TestGDSAsync(TestCase):
    hw_classification = HardwareClassification.CUDA

    def _require_gds(self):
        if not torch.cuda.gds.is_available():
            self.skipTest("cuFile/hipFile is not built in")
        # Probe only filesystem O_DIRECT support; do not hide backend I/O errors.
        with tempfile.NamedTemporaryFile() as file:
            try:
                fd = os.open(file.name, os.O_RDONLY | os.O_DIRECT)
            except OSError as error:
                import errno

                if error.errno in (errno.EINVAL, errno.EOPNOTSUPP):
                    self.skipTest("Temporary filesystem does not support O_DIRECT")
                raise
            os.close(fd)

    @parametrize("batch", [False, True])
    def test_gds_pending_io_owns_file_and_storage(self, device, batch):
        self._require_gds()
        src = [torch.arange(4096, device=device, dtype=torch.float32) + i for i in range(2)]
        dest = [torch.empty_like(t) for t in src]
        offsets = [0, src[0].nbytes]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with TemporaryFileName() as name:
            file = torch.cuda.gds.GdsFile(name, os.O_CREAT | os.O_RDWR)
            with torch.cuda.stream(stream):
                if batch:
                    writes = [file.save_storages([t.untyped_storage() for t in src], offsets)]
                else:
                    writes = [
                        file.save_storage_async(t.untyped_storage(), offset)
                        for t, offset in zip(src, offsets)
                    ]
            expected = [t.cpu() for t in src]
            del src
            file.close()
            for request in writes:
                request.wait()
                request.wait()
            file = torch.cuda.gds.GdsFile(name, os.O_RDONLY)
            with torch.cuda.stream(stream):
                if batch:
                    reads = [file.load_storages([t.untyped_storage() for t in dest], offsets)]
                else:
                    reads = [
                        file.load_storage_async(t.untyped_storage(), offset)
                        for t, offset in zip(dest, offsets)
                    ]
            file.close()
            for request in reads:
                request.wait()
            self.assertEqual(expected, [t.cpu() for t in dest])

    @parametrize("batch", [False, True])
    def test_gds_pending_io_short_read(self, device, batch):
        self._require_gds()
        dest = torch.empty(4096, device=device, dtype=torch.uint8)
        with TemporaryFileName() as name:
            file = torch.cuda.gds.GdsFile(name, os.O_CREAT | os.O_RDWR)
            # An empty file cannot satisfy the requested read.
            with self.assertRaisesRegex(RuntimeError, "GDS"):
                if batch:
                    request = file.load_storages([dest.untyped_storage()], [0])
                else:
                    request = file.load_storage_async(dest.untyped_storage())
                request.wait()
            file.close()

    @parametrize("batch", [False, True])
    def test_gds_resource_reuse(self, device, batch):
        self._require_gds()
        src = torch.arange(4096, device=device, dtype=torch.float32)
        dest = torch.empty_like(src)
        retained = []
        with TemporaryFileName() as name:
            file = torch.cuda.gds.GdsFile(name, os.O_CREAT | os.O_RDWR)
            try:
                for value in range(4):
                    src.fill_(value)
                    torch.cuda.synchronize()
                    file.save_storage(src.untyped_storage())
                    if batch:
                        request = file.load_storages([dest.untyped_storage()], [0])
                    else:
                        request = file.load_storage_async(dest.untyped_storage())
                    # Older results must not wait on or inspect reused state.
                    for old in retained:
                        old.wait()
                    request.wait()
                    retained.append(request)
                    self.assertEqual(src, dest)
                file.deregister_handle()
                file.register_handle()
                file.load_storage(dest.untyped_storage())
                self.assertEqual(src, dest)
            finally:
                file.close()
            for request in retained:
                request.wait()

    def test_gds_pending_io_validation(self, device):
        self._require_gds()
        storage = torch.empty(4096, device=device, dtype=torch.uint8).untyped_storage()
        with TemporaryFileName() as name:
            file = torch.cuda.gds.GdsFile(name, os.O_CREAT | os.O_RDWR)
            try:
                with self.assertRaisesRegex(ValueError, "equally sized"):
                    file.load_storages([storage], [])
                with self.assertRaisesRegex(RuntimeError, "nonnegative"):
                    file.load_storage_async(storage, -1)
                with self.assertRaisesRegex(RuntimeError, "at most 128"):
                    file.load_storages([storage] * 129, [0] * 129)
            finally:
                file.close()



instantiate_device_type_tests(TestGDSAsync, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
