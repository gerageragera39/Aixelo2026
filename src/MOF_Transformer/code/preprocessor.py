"""
MOFTransformer Preprocessor

Prepares raw MOF dataset for training by processing CIF files and creating
a tar archive with preprocessed data.

Author: MOFTransformer Team
Date: 2026-02-19
"""

import os
import sys
import json
import shutil
import tarfile
import argparse
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np

# Add MOFTransformer to path
sys.path.insert(0, str(Path(__file__).parent / "MOFTransformer"))

from moftransformer.utils.prepare_data import prepare_data
from moftransformer.utils.install_griday import install_griday


def verify_griday_installation() -> None:
    """
    Verify that GRIDAY is installed, install if necessary.
    
    GRIDAY is required for generating energy grid embeddings from CIF files.
    """
    try:
        from moftransformer import __root_dir__
        griday_path = os.path.join(__root_dir__, "libs/GRIDAY/scripts/grid_gen")
        if not os.path.exists(griday_path):
            print("GRIDAY not found. Installing GRIDAY...")
            install_griday()
    except ImportError as e:
        print(f"Error importing GRIDAY: {e}")
        print("Attempting to install GRIDAY...")
        install_griday()


def load_and_validate_id_prop(
    data_dir: Path,
    target_column: str
) -> Tuple[pd.DataFrame, str]:
    """
    Load id_prop.csv and validate the target column.
    
    The first column is assumed to contain CIF filenames (without extension).
    The target column is extracted for regression training.
    
    Parameters
    ----------
    data_dir : Path
        Path to the dataset directory containing id_prop.csv and raw/ folder.
    target_column : str
        Name of the column to use as regression target.
        
    Returns
    -------
    Tuple[pd.DataFrame, str]
        Filtered dataframe with only CIF ID and target columns,
        and the actual target column name (in case of case-insensitive match).
    """
    id_prop_path = data_dir / "id_prop.csv"
    
    if not id_prop_path.exists():
        raise FileNotFoundError(f"id_prop.csv not found at {id_prop_path}")
    
    # Read CSV - first row is header, first column is CIF filenames
    df = pd.read_csv(id_prop_path)
    
    if df.empty:
        raise ValueError("id_prop.csv is empty")
    
    if df.shape[1] < 2:
        raise ValueError(
            f"id_prop.csv must have at least 2 columns "
            f"(CIF filename and target value). Found {df.shape[1]} columns."
        )
    
    # Find target column (case-insensitive search)
    target_col_found = None
    for col in df.columns:
        if col.lower() == target_column.lower():
            target_col_found = col
            break
    
    if target_col_found is None:
        available_columns = list(df.columns)
        raise ValueError(
            f"Target column '{target_column}' not found in id_prop.csv. "
            f"Available columns: {available_columns}"
        )
    
    # Get the first column name (CIF filenames) - we don't care about its actual name
    first_column = df.columns[0]
    
    # Keep only CIF ID column and target column
    filtered_df = df[[first_column, target_col_found]].copy()
    
    # Rename columns to standard names for internal processing
    filtered_df.columns = ["cif_id", "target"]
    
    # Remove rows with NaN values
    initial_count = len(filtered_df)
    filtered_df = filtered_df.dropna()
    removed_count = initial_count - len(filtered_df)
    if removed_count > 0:
        print(f"Removed {removed_count} rows with NaN values")
    
    # Verify target values are numeric
    try:
        filtered_df["target"] = pd.to_numeric(filtered_df["target"])
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Target column must contain numeric values. Error: {e}"
        )
    
    print(f"Loaded {len(filtered_df)} samples from id_prop.csv")
    print(f"Target column: '{target_col_found}' (renamed internally to 'target')")
    print(f"Target statistics - Mean: {filtered_df['target'].mean():.4f}, "
          f"Std: {filtered_df['target'].std():.4f}, "
          f"Min: {filtered_df['target'].min():.4f}, "
          f"Max: {filtered_df['target'].max():.4f}")
    
    return filtered_df, target_col_found


def verify_cif_files(
    data_dir: Path,
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Verify that CIF files exist for all entries in the dataframe.
    
    Parameters
    ----------
    data_dir : Path
        Path to dataset directory containing raw/ folder.
    df : pd.DataFrame
        DataFrame with cif_id column.
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with only entries that have corresponding CIF files.
    """
    raw_dir = data_dir / "raw"
    
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw CIF files directory not found at {raw_dir}"
        )
    
    # Get all CIF files in raw directory
    cif_files = {f.stem: f.name for f in raw_dir.glob("*.cif")}
    print(f"Found {len(cif_files)} CIF files in raw/ directory")
    
    # Check which CIF IDs have corresponding files
    cif_ids = set(df["cif_id"].tolist())
    missing_in_cif = cif_ids - set(cif_files.keys())
    missing_in_prop = set(cif_files.keys()) - cif_ids
    
    if len(missing_in_cif) > 0:
        print(f"Warning: {len(missing_in_cif)} CIF IDs in id_prop.csv have no matching files")
        if len(missing_in_cif) <= 10:
            print(f"  Missing: {list(missing_in_cif)[:10]}")
    
    if len(missing_in_prop) > 0:
        print(f"Warning: {len(missing_in_prop)} CIF files not listed in id_prop.csv")
    
    # Keep only intersection
    valid_ids = cif_ids.intersection(set(cif_files.keys()))
    filtered_df = df[df["cif_id"].isin(valid_ids)].copy()
    
    print(f"Valid samples (present in both): {len(filtered_df)}")
    
    if len(filtered_df) == 0:
        raise ValueError(
            "No valid samples found after filtering. "
            "Please ensure CIF files match id_prop.csv entries."
        )
    
    return filtered_df


def create_raw_json(
    df: pd.DataFrame,
    output_path: Path,
    target_name: str
) -> None:
    """
    Create raw_{target}.json file mapping CIF IDs to target values.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with cif_id and target columns.
    output_path : Path
        Path to save the JSON file (should be in raw/ directory).
    target_name : str
        Name of the target for the downstream task name.
    """
    raw_data = {}
    for _, row in df.iterrows():
        cif_id = str(row["cif_id"])
        target_value = float(row["target"])
        raw_data[cif_id] = target_value
    
    json_path = output_path / f"raw_{target_name}.json"
    
    with open(json_path, "w") as f:
        json.dump(raw_data, f, indent=2)
    
    print(f"Created {json_path} with {len(raw_data)} entries")


def create_filtered_id_prop(
    df: pd.DataFrame,
    output_path: Path,
    original_header_name: str
) -> None:
    """
    Create filtered id_prop.csv with only CIF ID and target columns.
    
    The first column retains its original header name from the source file.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with cif_id and target columns.
    output_path : Path
        Path to save the filtered id_prop.csv.
    original_header_name : str
        Original name of the target column from source file.
    """
    # Restore original column names for output
    output_df = df.copy()
    output_df.columns = [df.columns[0], original_header_name]
    
    csv_path = output_path / "id_prop.csv"
    output_df.to_csv(csv_path, index=False)
    
    print(f"Created filtered {csv_path} with {len(output_df)} entries")


def run_data_preparation(
    raw_dir: Path,
    processed_dir: Path,
    target_name: str,
    train_fraction: float,
    test_fraction: float,
    seed: int
) -> None:
    """
    Run MOFTransformer's prepare_data to generate graph and grid embeddings.
    
    Parameters
    ----------
    raw_dir : Path
        Path to raw CIF files directory.
    processed_dir : Path
        Path to output processed data directory.
    target_name : str
        Name of the downstream target task.
    train_fraction : float
        Fraction of data for training.
    test_fraction : float
        Fraction of data for testing.
    seed : int
        Random seed for reproducibility.
    """
    print("\nStarting data preparation with MOFTransformer utilities...")
    print(f"  - Raw directory: {raw_dir}")
    print(f"  - Processed directory: {processed_dir}")
    print(f"  - Target: {target_name}")
    print(f"  - Train fraction: {train_fraction}")
    print(f"  - Test fraction: {test_fraction}")
    print(f"  - Seed: {seed}")
    
    prepare_data(
        root_cifs=raw_dir,
        root_dataset=processed_dir,
        downstream=target_name,
        train_fraction=train_fraction,
        test_fraction=test_fraction,
        seed=seed
    )
    
    # Verify output files
    expected_files = [
        processed_dir / f"train_{target_name}.json",
        processed_dir / f"val_{target_name}.json",
        processed_dir / f"test_{target_name}.json"
    ]
    
    print("\nVerifying processed files...")
    all_good = True
    for exp_file in expected_files:
        if exp_file.exists():
            size = exp_file.stat().st_size
            if size > 0:
                print(f"  ✓ {exp_file.name}: {size} bytes")
            else:
                print(f"  ✗ {exp_file.name}: EMPTY FILE")
                all_good = False
        else:
            print(f"  ✗ {exp_file.name}: MISSING")
            all_good = False
    
    if not all_good:
        raise RuntimeError(
            "Data preparation did not create valid training files. "
            "Check logs for details."
        )
    
    print("Data preparation completed successfully!")


def create_tar_archive(
    dataset_dir: Path,
    output_path: Path,
    target_name: str
) -> None:
    """
    Create tar archive with preprocessed dataset.

    Archive structure:
        dataset/
        └── processed/
            ├── train/
            ├── val/
            ├── test/
            ├── train_{target}.json
            ├── val_{target}.json
            └── test_{target}.json

    Parameters
    ----------
    dataset_dir : Path
        Path to the dataset directory to archive.
    output_path : Path
        Path for the output tar archive.
    target_name : str
        Target name for naming convention.
    """
    archive_name = f"{output_path}_{target_name}.tar"

    print(f"\nCreating tar archive: {archive_name}")

    with tarfile.open(archive_name, "w") as tar:
        # Add only the processed directory to the archive
        processed_dir = dataset_dir / "processed"
        tar.add(processed_dir, arcname="dataset/processed")
    
    # Get archive size
    archive_size = os.path.getsize(archive_name)
    size_mb = archive_size / (1024 * 1024)
    
    print(f"Archive created successfully!")
    print(f"  - Path: {archive_name}")
    print(f"  - Size: {size_mb:.2f} MB")


def preprocess_dataset(
    data_dir: str,
    target_column: str,
    output_name: str,
    train_fraction: float = 0.8,
    test_fraction: float = 0.1,
    seed: int = 42
) -> str:
    """
    Main preprocessing function.
    
    Parameters
    ----------
    data_dir : str
        Path to dataset directory containing id_prop.csv and raw/ folder.
    target_column : str
        Name of the target column in id_prop.csv to predict.
    output_name : str
        Base name for output tar archive.
    train_fraction : float, optional
        Fraction of data for training (default: 0.8).
    test_fraction : float, optional
        Fraction of data for testing (default: 0.1).
    seed : int, optional
        Random seed for reproducibility (default: 42).
        
    Returns
    -------
    str
        Path to the created tar archive.
    """
    print("=" * 60)
    print("MOFTransformer Preprocessor")
    print("=" * 60)
    
    # Convert to Path objects
    data_dir = Path(data_dir).resolve()
    output_name = Path(output_name).resolve()
    
    print(f"\nConfiguration:")
    print(f"  - Data directory: {data_dir}")
    print(f"  - Target column: {target_column}")
    print(f"  - Output name: {output_name}")
    
    # Verify GRIDAY installation
    print("\nStep 1: Verifying GRIDAY installation...")
    verify_griday_installation()
    
    # Load and validate id_prop.csv
    print("\nStep 2: Loading and validating id_prop.csv...")
    df, actual_target_name = load_and_validate_id_prop(data_dir, target_column)
    
    # Verify CIF files exist
    print("\nStep 3: Verifying CIF files...")
    df = verify_cif_files(data_dir, df)
    
    # Create working directory structure
    print("\nStep 4: Setting up working directory...")
    work_dir = data_dir / "preprocessed_work"
    raw_dir = work_dir / "raw"
    processed_dir = work_dir / "processed"
    
    # Clean up if exists
    if work_dir.exists():
        shutil.rmtree(work_dir)
    
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy CIF files to working raw directory
    print("Copying CIF files to working directory...")
    cif_ids = df["cif_id"].tolist()
    source_raw = data_dir / "raw"
    
    for cif_id in cif_ids:
        src = source_raw / f"{cif_id}.cif"
        dst = raw_dir / f"{cif_id}.cif"
        shutil.copy2(src, dst)
    
    print(f"Copied {len(cif_ids)} CIF files")
    
    # Create raw_{target}.json
    print("\nStep 5: Creating raw JSON file...")
    create_raw_json(df, raw_dir, actual_target_name)
    
    # Create filtered id_prop.csv
    print("\nStep 6: Creating filtered id_prop.csv...")
    create_filtered_id_prop(df, raw_dir, actual_target_name)
    
    # Run data preparation
    print("\nStep 7: Running data preparation (this may take a while)...")
    run_data_preparation(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        target_name=actual_target_name,
        train_fraction=train_fraction,
        test_fraction=test_fraction,
        seed=seed
    )
    
    # Create tar archive
    print("\nStep 8: Creating tar archive...")
    create_tar_archive(
        dataset_dir=work_dir,
        output_path=output_name,
        target_name=actual_target_name
    )
    
    # Cleanup working directory
    print("\nStep 9: Cleaning up working directory...")
    shutil.rmtree(work_dir)
    print("Working directory removed")
    
    print("\n" + "=" * 60)
    print("Preprocessing completed successfully!")
    print(f"Output archive: {output_name}_{actual_target_name}.tar")
    print("=" * 60)
    
    return str(output_name / f"{actual_target_name}.tar")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess MOF dataset for MOFTransformer training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python preprocessor.py --data-dir ./qmof_cif/ --target-column energy_per_atom --output-name qmof_preprocessed
  
  # Custom train/test split
  python preprocessor.py --data-dir ./qmof_cif/ --target-column bandgap --output-name qmof_bandgap --train-fraction 0.7 --test-fraction 0.15
  
  # With specific random seed
  python preprocessor.py --data-dir ./qmof_cif/ --target-column formation_energy --output-name qmof_formation --seed 123
        """
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to dataset directory containing id_prop.csv and raw/ folder"
    )
    
    parser.add_argument(
        "--target-column",
        type=str,
        required=True,
        help="Name of the target column in id_prop.csv to use for regression"
    )
    
    parser.add_argument(
        "--output-name",
        type=str,
        required=True,
        help="Base name for output tar archive (will be saved as {output-name}_{target}.tar)"
    )
    
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)"
    )
    
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.1,
        help="Fraction of data for testing (default: 0.1)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    try:
        preprocess_dataset(
            data_dir=args.data_dir,
            target_column=args.target_column,
            output_name=args.output_name,
            train_fraction=args.train_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed
        )
    except Exception as e:
        print(f"\nERROR: Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
