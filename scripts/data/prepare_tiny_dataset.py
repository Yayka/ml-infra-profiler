"""
Download roneneldan/TinyStories from HuggingFace and save as parquet files
in data/base_data/ (nanochat's data directory when NANOCHAT_BASE_DIR=<repo>/data).

nanochat's dataloader:
  - Reads all *.parquet files in base_data/, sorted alphabetically
  - Uses all files except the last as train split; last file as val split
  - Reads the 'text' column from each parquet row group

Usage (from repo root):
    .venv/bin/python scripts/data/prepare_tiny_dataset.py
"""

import pathlib

import datasets

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "base_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_ROWS = 50_000
VAL_ROWS = 5_000


def main():
    print(f"Saving parquet files to: {DATA_DIR}")

    print(f"Loading TinyStories train split ({TRAIN_ROWS:,} rows)...")
    train_ds = datasets.load_dataset("roneneldan/TinyStories", split="train")
    train_ds = train_ds.select(range(TRAIN_ROWS))

    # Verify the text column nanochat expects
    assert "text" in train_ds.column_names, (
        f"Expected 'text' column, got: {train_ds.column_names}"
    )

    train_path = DATA_DIR / "train_00.parquet"
    train_ds.to_parquet(str(train_path))
    print(f"  Saved {len(train_ds):,} rows → {train_path.name}")

    print(f"Loading TinyStories validation split ({VAL_ROWS:,} rows)...")
    val_ds = datasets.load_dataset("roneneldan/TinyStories", split="validation")
    val_ds = val_ds.select(range(min(VAL_ROWS, len(val_ds))))

    val_path = DATA_DIR / "val_00.parquet"
    val_ds.to_parquet(str(val_path))
    print(f"  Saved {len(val_ds):,} rows → {val_path.name}")

    print("\nDone. Files in data/base_data/:")
    for f in sorted(DATA_DIR.glob("*.parquet")):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}  ({size_mb:.1f} MB)")
    print(
        "\nnanochat will use train_00.parquet for training, "
        "val_00.parquet for validation."
    )


if __name__ == "__main__":
    main()
