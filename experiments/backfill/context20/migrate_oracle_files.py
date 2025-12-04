"""
Migrate Existing VAE Teacher Forcing Files to Oracle Subdirectory

Moves existing vae_tf_*.npz files from autoregressive/ to autoregressive/oracle/
to prepare for new sampling strategy parameter system.

Usage:
    python experiments/backfill/context20/migrate_oracle_files.py --dry-run  # Preview
    python experiments/backfill/context20/migrate_oracle_files.py            # Execute
"""

import argparse
import shutil
from pathlib import Path
import numpy as np

print("=" * 80)
print("VAE TEACHER FORCING FILES MIGRATION")
print("=" * 80)
print()

# Parse arguments
parser = argparse.ArgumentParser(description="Migrate vae_tf_*.npz files to oracle subdirectory")
parser.add_argument('--dry-run', action='store_true',
                    help='Preview changes without executing (default: False)')
args = parser.parse_args()

# Define paths
base_dir = Path("results/vae_baseline/predictions/autoregressive")
oracle_dir = base_dir / "oracle"

if not base_dir.exists():
    print(f"ERROR: Base directory does not exist: {base_dir}")
    print("Make sure you're running from repository root.")
    exit(1)

# Find all vae_tf_*.npz files in base directory
vae_tf_files = list(base_dir.glob("vae_tf_*.npz"))

if len(vae_tf_files) == 0:
    print(f"No vae_tf_*.npz files found in {base_dir}")
    print("Migration may have already been completed, or files are in a different location.")
    exit(0)

print(f"Found {len(vae_tf_files)} files to migrate:")
print()

# List files with sizes
total_size = 0
for f in sorted(vae_tf_files):
    size_mb = f.stat().st_size / (1024**2)
    total_size += size_mb
    print(f"  {f.name:35} {size_mb:8.1f} MB")

print()
print(f"Total size: {total_size:.1f} MB")
print()

if args.dry_run:
    print("=" * 80)
    print("DRY RUN MODE - No changes will be made")
    print("=" * 80)
    print()
    print("Actions that would be performed:")
    print(f"  1. Create directory: {oracle_dir}")
    print(f"  2. Move {len(vae_tf_files)} files to oracle/ subdirectory")
    print()
    print("To execute migration, run without --dry-run flag")
    exit(0)

# Proceed with migration
print("=" * 80)
print("EXECUTING MIGRATION")
print("=" * 80)
print()

# Create oracle directory
print(f"Creating directory: {oracle_dir}")
oracle_dir.mkdir(parents=True, exist_ok=True)
print("  ✓ Directory created")
print()

# Move files
print("Moving files...")
failed_files = []
for f in sorted(vae_tf_files):
    dest = oracle_dir / f.name
    try:
        # Move file
        shutil.move(str(f), str(dest))

        # Verify file integrity (can still load NPZ)
        try:
            data = np.load(dest, allow_pickle=True)
            data.close()
            print(f"  ✓ {f.name:35} → oracle/")
        except Exception as e:
            print(f"  ✗ {f.name:35} MOVED but failed validation: {e}")
            failed_files.append((f.name, str(e)))
    except Exception as e:
        print(f"  ✗ {f.name:35} FAILED to move: {e}")
        failed_files.append((f.name, str(e)))

print()

# Summary
print("=" * 80)
print("MIGRATION SUMMARY")
print("=" * 80)
print()

successful = len(vae_tf_files) - len(failed_files)
print(f"Successfully migrated: {successful}/{len(vae_tf_files)} files")

if failed_files:
    print(f"Failed: {len(failed_files)} files")
    print()
    print("Failed files:")
    for fname, error in failed_files:
        print(f"  - {fname}: {error}")
    print()
    print("⚠ WARNING: Some files failed to migrate")
    exit(1)
else:
    print()
    print("✓ All files migrated successfully")
    print()
    print("Next steps:")
    print("  1. Run validation: python experiments/backfill/context20/validate_vae_tf_sequences.py --sampling_mode oracle")
    print("  2. Generate prior sequences: bash experiments/backfill/context20/run_generate_all_tf_sequences.sh prior")
    print()
