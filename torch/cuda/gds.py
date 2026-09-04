from __future__ import annotations

import os
from collections.abc import Callable, Sequence

import torch
from torch.types import Storage


__all__: list[str] = [
    "is_available",
    "gds_register_buffer",
    "gds_deregister_buffer",
    "GdsFile",
    "GdsIO",
]


def _dummy_fn(name: str) -> Callable:
    def fn(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError(f"torch._C.{name} is not supported on this platform")

    return fn


def is_available() -> bool:
    r"""Return ``True`` if GPUDirect Storage (GDS) support is built in.

    This requires a Linux build with ``USE_CUFILE`` enabled (the default) and
    with cuFile (CUDA) or hipFile (ROCm) available at build time. See
    :ref:`rocm-gds` for the ROCm requirements.
    """
    return torch._C._has_gds


# Without GDS built in, install dummy stubs so the API raises a clear error.
if not is_available():
    if hasattr(torch._C, "_gds_deregister_buffer"):
        raise AssertionError(
            "_gds_deregister_buffer exists but _gds_register_buffer does not"
        )
    if hasattr(torch._C, "_gds_register_handle"):
        raise AssertionError(
            "_gds_register_handle exists but _gds_register_buffer does not"
        )
    if hasattr(torch._C, "_gds_load_storage"):
        raise AssertionError(
            "_gds_load_storage exists but _gds_register_buffer does not"
        )
    if hasattr(torch._C, "_gds_save_storage"):
        raise AssertionError(
            "_gds_save_storage exists but _gds_register_buffer does not"
        )
    torch._C.__dict__["_gds_submit"] = _dummy_fn("_gds_submit")
    # Define functions
    torch._C.__dict__["_gds_register_buffer"] = _dummy_fn("_gds_register_buffer")
    torch._C.__dict__["_gds_deregister_buffer"] = _dummy_fn("_gds_deregister_buffer")
    torch._C.__dict__["_gds_register_handle"] = _dummy_fn("_gds_register_handle")
    torch._C.__dict__["_gds_load_storage"] = _dummy_fn("_gds_load_storage")
    torch._C.__dict__["_gds_save_storage"] = _dummy_fn("_gds_save_storage")


def gds_register_buffer(s: Storage) -> None:
    """Registers a storage on a CUDA device as a GDS buffer.

    This is a wrapper around ``cuFileBufRegister`` (CUDA) / ``hipFileBufRegister`` (ROCm).

    Example::

        >>> # xdoctest: +SKIP("gds filesystem requirements")
        >>> src = torch.randn(1024, device="cuda")
        >>> s = src.untyped_storage()
        >>> gds_register_buffer(s)

    Args:
        s (Storage): Buffer to register.
    """
    torch._C._gds_register_buffer(s)


def gds_deregister_buffer(s: Storage) -> None:
    """Deregisters a previously registered GDS buffer.

    This is a wrapper around ``cuFileBufDeregister`` (CUDA) / ``hipFileBufDeregister`` (ROCm).

    Example::

        >>> # xdoctest: +SKIP("gds filesystem requirements")
        >>> src = torch.randn(1024, device="cuda")
        >>> s = src.untyped_storage()
        >>> gds_register_buffer(s)
        >>> gds_deregister_buffer(s)

    Args:
        s (Storage): Buffer to register.
    """
    torch._C._gds_deregister_buffer(s)



class GdsIO:
    """Pending cuFile/hipFile I/O returned by :class:`GdsFile`.

    Call :meth:`wait` before accessing a read destination or modifying a write
    source. The operation retains its storages and file descriptor. Dropping
    it without waiting is safe but blocks; use wait() to observe I/O errors.
    Do not resize or explicitly deregister buffers while I/O is pending.
    """

    def __init__(self, request: torch._C._GdsIO, stream: torch.cuda.Stream):
        self._request = request
        self._stream = stream

    def wait(self) -> None:
        """Wait for completion and raise on I/O errors, including short reads."""
        self._request.wait()

    def __del__(self) -> None:
        # Destroy the native request before releasing the stream reference.
        del self._request


class GdsFile:
    r"""Wrapper around a file registered with the GPUDirect Storage (GDS) driver.

    cuFile (CUDA) and hipFile (ROCm) are file-like interfaces to the GDS API.

    See the `cufile docs <https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufile-io-api>`_
    and the `hipFile docs <https://rocm.docs.amd.com/projects/hipFile/en/latest/>`_
    for more details, and :ref:`rocm-gds` for ROCm specifics.

    Args:
        filename (str): Name of the file to open.
        flags (int): Flags to pass to ``os.open`` when opening the file. ``os.O_DIRECT`` will
            be added automatically.

    Example::

        >>> # xdoctest: +SKIP("gds filesystem requirements")
        >>> src1 = torch.randn(1024, device="cuda")
        >>> src2 = torch.randn(2, 1024, device="cuda")
        >>> file = torch.cuda.gds.GdsFile(f, os.O_CREAT | os.O_RDWR)
        >>> file.save_storage(src1.untyped_storage(), offset=0)
        >>> file.save_storage(src2.untyped_storage(), offset=src1.nbytes)
        >>> dest1 = torch.empty(1024, device="cuda")
        >>> dest2 = torch.empty(2, 1024, device="cuda")
        >>> file.load_storage(dest1.untyped_storage(), offset=0)
        >>> file.load_storage(dest2.untyped_storage(), offset=src1.nbytes)
        >>> torch.equal(src1, dest1)
        True
        >>> torch.equal(src2, dest2)
        True

    """

    def __init__(self, filename: str, flags: int):
        self.filename = filename
        self.flags = flags
        self.fd = os.open(filename, flags | os.O_DIRECT)  # type: ignore[attr-defined]
        self.handle: torch._C._GdsHandle | None = None
        self.register_handle()

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """Deregisters the handle if needed and closes the file descriptor.

        Calling this more than once is a no-op.
        """
        if self.fd < 0:
            return
        try:
            if self.handle is not None:
                self.deregister_handle()
        finally:
            os.close(self.fd)
            self.fd = -1

    def register_handle(self) -> None:
        """Registers file descriptor to the GDS driver.

        This is a wrapper around ``cuFileHandleRegister`` (CUDA) / ``hipFileHandleRegister`` (ROCm).
        """
        if self.handle is not None:
            raise AssertionError("Cannot register a handle that is already registered.")
        self.handle = torch._C._gds_register_handle(self.fd)

    def deregister_handle(self) -> None:
        """Deregisters file descriptor from the GDS driver.

        This is a wrapper around ``cuFileHandleDeregister`` (CUDA) / ``hipFileHandleDeregister`` (ROCm).
        """
        if self.handle is None:
            raise AssertionError("Cannot deregister a handle that is not registered.")
        self.handle = None

    def load_storage(self, storage: Storage, offset: int = 0) -> None:
        """Loads data from the file into the storage.

        This is a wrapper around ``cuFileRead`` (CUDA) / ``hipFileRead`` (ROCm).
        ``storage.nbytes()`` of data will be loaded from the file at ``offset``
        into the storage.

        Args:
            storage (Storage): Storage to load data into.
            offset (int, optional): Offset into the file to start loading from. (Default: 0)
        """
        if self.handle is None:
            raise AssertionError("Cannot load data from a file that is not registered.")
        torch._C._gds_load_storage(self.handle, storage, offset)

    def save_storage(self, storage: Storage, offset: int = 0) -> None:
        """Saves data from the storage into the file.

        This is a wrapper around ``cuFileWrite`` (CUDA) / ``hipFileWrite`` (ROCm).
        All bytes of the storage will be written to the file at ``offset``.

        Args:
            storage (Storage): Storage to save data from.
            offset (int, optional): Offset into the file to start saving to. (Default: 0)
        """
        if self.handle is None:
            raise AssertionError("Cannot save data to a file that is not registered.")
        torch._C._gds_save_storage(self.handle, storage, offset)

    def _submit(
        self,
        storages: Sequence[Storage],
        offsets: Sequence[int],
        *,
        write: bool,
        batch: bool,
        stream: torch.cuda.Stream | None = None,
    ) -> GdsIO:
        if self.handle is None:
            raise RuntimeError("Cannot submit I/O on an unregistered GdsFile")
        storages, offsets = list(storages), list(offsets)
        if not storages or len(storages) != len(offsets):
            raise ValueError("Expected equally sized, nonempty storages and offsets")
        device = storages[0].device
        if device.type != "cuda":
            raise ValueError("GDS requires CUDA/HIP storage")
        if stream is None:
            stream = torch.cuda.current_stream(device)
        if stream.device != device:
            raise ValueError("GDS stream and storage must be on the same device")
        with torch.cuda.stream(stream):
            request = torch._C._gds_submit(self.handle, storages, offsets, write, batch)
        return GdsIO(request, stream)

    def load_storage_async(
        self, storage: Storage, offset: int = 0, *,
        stream: torch.cuda.Stream | None = None,
    ) -> GdsIO:
        """Enqueue a read using cuFileReadAsync/hipFileReadAsync.

        Uses the current stream unless specified. Call wait() to check I/O
        success before consuming the storage, even for stream-ordered work.
        """
        return self._submit([storage], [offset], write=False, batch=False, stream=stream)

    def save_storage_async(
        self, storage: Storage, offset: int = 0, *,
        stream: torch.cuda.Stream | None = None,
    ) -> GdsIO:
        """Enqueue a stream-ordered write using cuFileWriteAsync/hipFileWriteAsync."""
        return self._submit([storage], [offset], write=True, batch=False, stream=stream)

    def load_storages(
        self, storages: Sequence[Storage], offsets: Sequence[int],
    ) -> GdsIO:
        """Submit up to 128 reads using the native batch API.

        All storages must be on one device and destination ranges must not
        overlap. Submission synchronizes the current stream on that device;
        batch I/O itself is not stream ordered. Call wait() before using results.
        """
        return self._submit(storages, offsets, write=False, batch=True)

    def save_storages(
        self, storages: Sequence[Storage], offsets: Sequence[int],
    ) -> GdsIO:
        """Submit up to 128 writes; file ranges must not overlap.

        Synchronizes producer work on the current stream before submitting.
        Call wait() before modifying sources or relying on the file contents.
        """
        return self._submit(storages, offsets, write=True, batch=True)
