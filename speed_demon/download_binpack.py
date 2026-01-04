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


def _write_binpack_limited(
    reader,
    output_file,
    chunk_mb: int,
    max_bytes: int,
    start_time: Optional[float] = None,
    written_offset: int = 0,
) -> int:
    pending = bytearray()
    written = 0
    header = None
    chunk_size = None
    if start_time is None:
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
                    _print_progress(written_offset + written, start_time)
                    return written
                output_file.write(header)
                output_file.write(pending[:chunk_size])
                del pending[:chunk_size]
                written += 8 + chunk_size
                header = None
                chunk_size = None
                _print_progress(written_offset + written, start_time)

        if not data:
            break

    _print_progress(written_offset + written, start_time)
    return written


def _load_urls(url_args, urls_file: Optional[str]) -> list[str]:
    urls: list[str] = []
    if url_args:
        for entry in url_args:
            if not entry:
                continue
            parts = [part.strip() for part in entry.split(",") if part.strip()]
            urls.extend(parts)
    if urls_file:
        with open(urls_file, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                urls.append(line)
    if not urls:
        urls.append(DEFAULT_URL)
    return urls


def download_and_decompress(
    urls: list[str],
    output_path: str,
    chunk_mb: int = 1,
    max_gb: Optional[float] = None,
    append: bool = False,
) -> None:
    print("--- DATASET ACQUISITION ---")
    if len(urls) == 1:
        print(f"Source: {urls[0]}")
    else:
        print(f"Sources: {len(urls)}")
        for idx, url in enumerate(urls, start=1):
            print(f"  {idx:02d}: {url}")
    print(f"Target: {output_path}")

    if os.path.exists(output_path) and not append:
        print(f"File already exists: {output_path}")
        print("Delete it or pass --append to extend it.")
        return

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    max_bytes = int(max_gb * (1024**3)) if max_gb is not None else None
    mode = "ab" if append else "wb"
    total_written = 0
    if append and os.path.exists(output_path):
        total_written = os.path.getsize(output_path)
        if max_bytes is not None and total_written >= max_bytes:
            print(
                f"Output already has {total_written / (1024**3):.2f} GB; max-gb reached."
            )
            return

    start_time = time.time()
    with open(output_path, mode) as output_file:
        for idx, url in enumerate(urls, start=1):
            remaining = None if max_bytes is None else max_bytes - total_written
            if remaining is not None and remaining <= 0:
                print("\nReached max-gb limit.")
                break
            print(f"Downloading and decompressing ({idx}/{len(urls)}): {url}")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(response.raw) as reader:
                if remaining is not None:
                    written = _write_binpack_limited(
                        reader,
                        output_file,
                        chunk_mb,
                        remaining,
                        start_time=start_time,
                        written_offset=total_written,
                    )
                    total_written += written
                else:
                    while True:
                        chunk = reader.read(chunk_mb * 1024 * 1024)
                        if not chunk:
                            break
                        output_file.write(chunk)
                        total_written += len(chunk)
                        _print_progress(total_written, start_time)

    print("\nDone. Dataset is ready.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NNUE binpack dataset")
    parser.add_argument(
        "--url",
        action="append",
        help="Dataset URL (repeatable or comma-separated).",
    )
    parser.add_argument(
        "--urls-file",
        type=str,
        default=None,
        help="Path to a newline-delimited list of URLs.",
    )
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
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing output file instead of skipping it.",
    )
    args = parser.parse_args()

    urls = _load_urls(args.url, args.urls_file)
    download_and_decompress(
        urls,
        args.output,
        args.chunk_mb,
        args.max_gb,
        append=args.append,
    )


if __name__ == "__main__":
    main()
