import argparse
import os
import sys
import time

import requests
import zstandard as zstd

DEFAULT_URL = (
    "https://huggingface.co/datasets/linrock/test80-2024/resolve/main/"
    "test80-2024-01-jan-2tb7p.min-v2.v6.binpack.zst"
)


def download_and_decompress(url: str, output_path: str, chunk_mb: int = 1) -> None:
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
    start_time = time.time()
    downloaded = 0

    with open(output_path, "wb") as output_file:
        with dctx.stream_reader(response.raw) as reader:
            while True:
                chunk = reader.read(chunk_mb * 1024 * 1024)
                if not chunk:
                    break
                output_file.write(chunk)
                downloaded += len(chunk)

                elapsed = time.time() - start_time
                speed = downloaded / max(elapsed, 0.001)
                sys.stdout.write(
                    f"\r  Saved: {downloaded / (1024**3):.2f} GB"
                    f" | Speed: {speed / (1024**2):.1f} MB/s"
                )
                sys.stdout.flush()

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
    args = parser.parse_args()

    download_and_decompress(args.url, args.output, args.chunk_mb)


if __name__ == "__main__":
    main()
