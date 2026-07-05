#!/usr/bin/env python
"""
Utility script to prepare DAIC-WOZ dataset after download.
"""

import os
import subprocess
from pathlib import Path


def check_prerequisites():
    """
    Check if all required packages are installed.
    """
    print("Checking prerequisites...")
    
    required_packages = [
        'librosa',
        'opencv-python',
        'torch',
        'pandas',
        'numpy'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True


def check_daic_woz_structure():
    """
    Check if DAIC-WOZ dataset is properly extracted.
    """
    print("\nChecking DAIC-WOZ structure...")
    
    dataset_dir = Path('data/datasets/DAIC-WOZ')
    
    if not dataset_dir.exists():
        print(f"✗ Dataset directory not found: {dataset_dir}")
        print(f"Please download from: https://dcapswoz.ict.usc.edu/")
        return False
    
    required_dirs = ['train_data', 'dev_data', 'test_data']
    required_files = ['labels.csv']
    
    for d in required_dirs:
        path = dataset_dir / d
        if path.exists():
            count = len(list(path.glob('*')))
            print(f"✓ {d} ({count} samples)")
        else:
            print(f"✗ Missing directory: {d}")
            return False
    
    for f in required_files:
        path = dataset_dir / f
        if path.exists():
            print(f"✓ {f}")
        else:
            print(f"✗ Missing file: {f}")
            return False
    
    return True


def run_preprocessing():
    """
    Run data preprocessing.
    """
    print("\nRunning preprocessing...")
    
    try:
        result = subprocess.run(
            ['python', 'src/data_processing/data_preprocessor.py', '--dataset', 'daic-woz'],
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return False
        
        return True
    
    except Exception as e:
        print(f"✗ Preprocessing failed: {e}")
        return False


def verify_preprocessing():
    """
    Verify that preprocessing completed successfully.
    """
    print("\nVerifying preprocessing...")
    
    processed_dir = Path('data/processed/DAIC-WOZ')
    
    if not processed_dir.exists():
        print(f"✗ Processed data directory not found: {processed_dir}")
        return False
    
    # Count feature files
    feature_files = list(processed_dir.glob('*/features.json'))
    print(f"✓ Found {len(feature_files)} feature files")
    
    # Check statistics
    stats_file = processed_dir / 'statistics.json'
    if stats_file.exists():
        print(f"✓ Statistics file exists")
    
    # Check splits
    splits_file = processed_dir / 'splits.json'
    if splits_file.exists():
        print(f"✓ Splits file exists")
    
    return len(feature_files) > 0


def main():
    """
    Main preparation function.
    """
    print("="*80)
    print("DAIC-WOZ Dataset Preparation")
    print("="*80)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Please install missing packages first")
        return False
    
    # Check dataset structure
    if not check_daic_woz_structure():
        print("\n❌ DAIC-WOZ dataset not properly downloaded/extracted")
        return False
    
    # Run preprocessing
    if not run_preprocessing():
        print("\n❌ Preprocessing failed")
        return False
    
    # Verify preprocessing
    if not verify_preprocessing():
        print("\n❌ Preprocessing verification failed")
        return False
    
    print("\n" + "="*80)
    print("✅ DAIC-WOZ preparation completed successfully!")
    print("="*80)
    print("\nNext steps:")
    print("1. Run training: python src/training/train.py")
    print("2. Run inference: python main.py --video <video_path>")
    print("3. Check results: ls results/")
    
    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
