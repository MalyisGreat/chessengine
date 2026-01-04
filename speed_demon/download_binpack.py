import argparse
import os
import sys
import time
from typing import Optional

import requests
import zstandard as zstd

DEFAULT_URL = (
    "https://huggingface.co/datasets/linrock/test80-2024/resolve/main/"
    "test80-2024-01-jan-2tb7p.min-v2.v6.binpack.zst"
)


def _print_progress(written: int, start_time: float) -> None:
    elapsed = time.time() - start_time
    speed = written / max(elapsed, 0.001)
    sys.stdout.write(
        f"\r  Saved: {written / (1024**3):.2f} GB | Speed: {speed / (1024**2):.1f} MB/s"
    )
    sys.stdout.flush()


def _write_binpack_limited(reader, output_file, chunk_mb: int, max_bytes: int) -> int:
    pending = bytearray()
    written = 0
    header = None
    chunk_size = None
    start_time = time.time()

    while True:
        data = reader.read(chunk_mb * 1024 * 1024)
        if data:
            pending.extend(data)
        elif not pending:
            break

        while True:
            if header is None:
                if len(pending) < 8:
                    break
                header = bytes(pending[:8])
                del pending[:8]
                if header[:4] != b"BINP":
                    raise ValueError("Invalid binpack header; expected BINP.")
                chunk_size = int.from_bytes(header[4:8], "little")

            if header is not None:
                if len(pending) < chunk_size:
                    break
                if written + 8 + chunk_size > max_bytes:
                    _print_progress(written, start_time)
                    return written
                output_file.write(header)
                output_file.write(pending[:chunk_size])
                del pending[:chunk_size]
                written += 8 + chunk_size
                header = None
                chunk_size = None
                _print_progress(written, start_time)

        if not data:
            break

    _print_progress(written, start_time)
    return written


def download_and_decompress(
    url: str,
    output_path: str,
    chunk_mb: int = 1,
    max_gb: Optional[float] = None,
) -> None:
    print("--- DATASET ACQUISITION ---")
    print(f"Source: {url}")
    print(f"Target: {output_path}")

    if os.path.exists(output_path):
        print(f"File already exists: {output_path}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    print("Downloading and decompressing...")
    dctx = zstd.ZstdDecompressor()
    with open(output_path, "wb") as output_file:
        with dctx.stream_reader(response.raw) as reader:
            if max_gb is not None:
                max_bytes = int(max_gb * (1024**3))
                _write_binpack_limited(reader, output_file, chunk_mb, max_bytes)
            else:
                start_time = time.time()
                downloaded = 0
                while True:
                    chunk = reader.read(chunk_mb * 1024 * 1024)
                    if not chunk:
                        break
                    output_file.write(chunk)
                    downloaded += len(chunk)
                    _print_progress(downloaded, start_time)

    print("\nDone. Dataset is ready.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NNUE binpack dataset")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="Dataset URL")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("data", "binpack", "training_data.binpack"),
        help="Output .binpack path",
    )
    parser.add_argument(
        "--chunk-mb",
        type=int,
        default=1,
        help="Chunk size in MB for streaming decompression",
    )
    parser.add_argument(
        "--max-gb",
        type=float,
        default=None,
        help="Stop after writing this many GB (chunk-aligned).",
    )
    args = parser.parse_args()

    download_and_decompress(args.url, args.output, args.chunk_mb, args.max_gb)


if __name__ == "__main__":
    main()
