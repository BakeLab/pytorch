"""Measure GDS submission, completion, and resource cleanup separately."""

import argparse
import hashlib
import json
import os
import time

import torch
from torch.cuda.gds import GdsFile

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--mode", choices=("sync", "batch", "async"), required=True)
    parser.add_argument("--chunk-mib", type=int, default=16)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--waves", type=int, default=8)
    args = parser.parse_args()
    if min(args.chunk_mib, args.width, args.waves) <= 0 or args.offset < 0:
        parser.error("sizes must be positive and offset nonnegative")
    size = args.chunk_mib * 1024**2
    buffers = [torch.empty(size, dtype=torch.uint8, device="cuda") for _ in range(args.width)]
    storages = [tensor.untyped_storage() for tensor in buffers]
    streams = [torch.cuda.Stream() for _ in buffers]
    torch.cuda.synchronize()
    start = time.perf_counter()
    file = GdsFile(args.path, os.O_RDONLY)
    setup = time.perf_counter() - start
    samples = []
    try:
        for wave in range(args.waves):
            offsets = [args.offset + (wave * args.width + i) * size for i in range(args.width)]
            torch.cuda.synchronize()
            requests = []
            start = time.perf_counter()
            if args.mode == "sync":
                for storage, offset in zip(storages, offsets):
                    file.load_storage(storage, offset)
            elif args.mode == "batch":
                requests.append(file.load_storages(storages, offsets))
            else:
                requests.extend(
                    file.load_storage_async(storage, offset, stream=stream)
                    for storage, offset, stream in zip(storages, offsets, streams)
                )
            submitted = time.perf_counter()
            for request in requests:
                request.wait()
            if requests:
                del request
            waited = time.perf_counter()
            requests.clear()
            cleaned = time.perf_counter()
            torch.cuda.synchronize()
            synchronized = time.perf_counter()
            samples.append(dict(submit=submitted - start, wait=waited - submitted,
                                cleanup=cleaned - waited, synchronize=synchronized - cleaned))
        for tensor, offset in zip(buffers, offsets):
            with open(args.path, "rb") as source:
                source.seek(offset)
                expected = hashlib.sha256(source.read(size)).digest()
            if hashlib.sha256(tensor.cpu().numpy().tobytes()).digest() != expected:
                raise RuntimeError("GDS data mismatch")
    finally:
        start = time.perf_counter()
        file.close()
        close = time.perf_counter() - start
    print(json.dumps(dict(version=torch.__version__, mode=args.mode, bytes=size * args.width * args.waves,
                         setup=setup, close=close, samples=samples, verified=True,
                         totals={key: sum(sample[key] for sample in samples) for key in samples[0]})))

if __name__ == "__main__":
    main()
