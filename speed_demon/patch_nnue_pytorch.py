import argparse
from pathlib import Path


def _insert_after_l1_arg(text: str) -> str:
    if "--l2" in text or "--l3" in text:
        return text

    marker = "parser.add_argument(\"--l1\""
    lines = text.splitlines()
    out_lines = []
    inserted = False
    for line in lines:
        out_lines.append(line)
        if not inserted and marker in line:
            indent = line[: len(line) - len(line.lstrip())]
            out_lines.append(f"{indent}parser.add_argument(\"--l2\", type=int, default=M.ModelConfig().L2)")
            out_lines.append(f"{indent}parser.add_argument(\"--l3\", type=int, default=M.ModelConfig().L3)")
            inserted = True
    return "\n".join(out_lines)


def _patch_model_config_usage(text: str) -> str:
    if "L2=args.l2" in text and "L3=args.l3" in text:
        return text
    return text.replace(
        "M.ModelConfig(L1=args.l1)",
        "M.ModelConfig(L1=args.l1, L2=args.l2, L3=args.l3)",
    )


def _patch_native_aliases(text: str) -> str:
    lines = text.splitlines()
    out_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("type SparseBatchPtr ="):
            indent = line[: len(line) - len(line.lstrip())]
            out_lines.append(f"{indent}SparseBatchPtr = ctypes.POINTER(SparseBatch)")
            continue
        if stripped.startswith("type FenBatchPtr ="):
            indent = line[: len(line) - len(line.lstrip())]
            out_lines.append(f"{indent}FenBatchPtr = ctypes.POINTER(FenBatch)")
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


def _patch_halfkp_psqt(text: str) -> str:
    if "Not supported yet. See HalfKA" not in text:
        return text

    replacement = (
        "    def get_initial_psqt_features(self) -> list[int]:\n"
        "        return [0] * self.num_features\n"
    )

    text = text.replace(
        "    def get_initial_psqt_features(self):\n"
        "        raise Exception(\"Not supported yet. See HalfKA\")\n",
        replacement,
    )
    return text


def _patch_csv_logger(text: str) -> str:
    if "CSVLogger" in text and "csv_logger" in text:
        return text
    text = text.replace(
        "tb_logger = pl_loggers.TensorBoardLogger(logdir)",
        "tb_logger = pl_loggers.TensorBoardLogger(logdir)\n"
        "    csv_logger = pl_loggers.CSVLogger(logdir)",
    )
    text = text.replace(
        "logger=tb_logger,",
        "logger=[tb_logger, csv_logger],",
    )
    return text


def _patch_matmul_precision(text: str) -> str:
    if "TORCH_MATMUL_PRECISION" in text:
        return text
    marker = "import torch"
    lines = text.splitlines()
    out_lines = []
    inserted = False
    for line in lines:
        out_lines.append(line)
        if not inserted and line.strip() == marker:
            out_lines.append("")
            out_lines.append("if os.environ.get(\"TORCH_MATMUL_PRECISION\"):")
            out_lines.append("    try:")
            out_lines.append(
                "        torch.set_float32_matmul_precision(os.environ[\"TORCH_MATMUL_PRECISION\"])"
            )
            out_lines.append("    except Exception:")
            out_lines.append("        pass")
            out_lines.append("if os.environ.get(\"TORCH_TF32\") == \"1\":")
            out_lines.append("    torch.backends.cuda.matmul.allow_tf32 = True")
            out_lines.append("    torch.backends.cudnn.allow_tf32 = True")
            inserted = True
    return "\n".join(out_lines)


def patch_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    updated = original
    if path.name in ("train.py", "serialize.py"):
        updated = _insert_after_l1_arg(updated)
        updated = _patch_model_config_usage(updated)
    if path.name == "train.py":
        updated = _patch_csv_logger(updated)
        updated = _patch_matmul_precision(updated)
    if path.name == "_native.py":
        updated = _patch_native_aliases(updated)
    if path.name == "halfkp.py":
        updated = _patch_halfkp_psqt(updated)
    if updated == original:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch nnue-pytorch for L2/L3 args")
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Path to nnue-pytorch checkout",
    )
    args = parser.parse_args()

    repo = Path(args.repo)
    train_path = repo / "train.py"
    serialize_path = repo / "serialize.py"
    native_path = repo / "data_loader" / "_native.py"
    halfkp_path = repo / "model" / "features" / "halfkp.py"

    patched_any = False
    for target in (train_path, serialize_path, native_path, halfkp_path):
        if not target.exists():
            raise FileNotFoundError(f"Missing {target}")
        if patch_file(target):
            patched_any = True

    if patched_any:
        print("Patched nnue-pytorch for L2/L3 support.")
    else:
        print("nnue-pytorch already patched.")


if __name__ == "__main__":
    main()
