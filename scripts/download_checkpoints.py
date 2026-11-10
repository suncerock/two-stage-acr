"""Download the pretrained checkpoints from Zenodo into ``ckpt/`` (cross-platform, standard library only)."""

import argparse
import urllib.request
from pathlib import Path

ZENODO_RECORD = "24103648"
CHECKPOINTS = [
    "pretrained_mpe",
    "pretrained_chord",
    "crnn_pop",
    "conformer_pop",
    "crnn_classical",
    "conformer_classical",
    "pretrained_chord_withN",
    "crnn_pop_withN",
    "conformer_pop_withN",
    "crnn_classical_withN",
    "conformer_classical_withN",
]


def download(name: str, output_dir: Path, overwrite: bool) -> None:
    target = output_dir / f"{name}.ckpt"
    if target.exists() and not overwrite:
        print(f"Skipping {target} (exists, use --overwrite to replace)")
        return

    url = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{name}.ckpt?download=1"
    print(f"Downloading {url}")
    tmp_target = target.with_suffix(".ckpt.part")
    try:
        urllib.request.urlretrieve(url, tmp_target)
        tmp_target.replace(target)
    finally:
        tmp_target.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the pretrained checkpoints from Zenodo.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "ckpt",
        help="Target folder (default: ckpt/ in the repository root).",
    )
    parser.add_argument(
        "--only", nargs="+", choices=CHECKPOINTS, metavar="NAME",
        help="Download only these checkpoints (default: all).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-download files that already exist.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in args.only or CHECKPOINTS:
        download(name, args.output_dir, args.overwrite)


if __name__ == "__main__":
    main()
