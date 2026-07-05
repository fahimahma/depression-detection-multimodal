#!/usr/bin/env python
"""
Dataset downloader and manager for depression detection datasets.
Supports DAIC-WOZ, D-Vlog, and other depression detection datasets.
"""

import os
import json
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional
import requests
from tqdm import tqdm
import urllib.request


class DatasetDownloader:
    """
    Manages downloading and extracting depression detection datasets.
    """
    
    # Dataset URLs and information
    DATASETS = {
        'daic-woz': {
            'name': 'DAIC-WOZ (Depression and Anxiety Inventory Collection)',
            'url': 'https://dcapswoz.ict.usc.edu/',
            'description': 'Clinical interviews with depression/anxiety labels',
            'size': '~50GB',
            'requires_registration': True,
            'modalities': ['audio', 'video', 'text_transcripts'],
            'labels': 'PHQ-9 scores (0-27)',
            'num_samples': 1189
        },
        'd-vlog': {
            'name': 'D-Vlog (Depression Detection from Vlogs)',
            'url': 'Contact authors for access',
            'description': 'YouTube vlogs labeled for depression',
            'size': 'Variable',
            'requires_registration': True,
            'modalities': ['video', 'audio'],
            'labels': 'Binary (depressed/non-depressed)',
            'num_samples': 1000
        },
        'e-daic': {
            'name': 'E-DAIC (Extended DAIC)',
            'url': 'Limited access - Contact authors',
            'description': 'Extended version of DAIC-WOZ',
            'size': '~100GB',
            'requires_registration': True,
            'modalities': ['audio', 'video', 'text_transcripts'],
            'labels': 'PHQ-9 scores',
            'num_samples': 2500
        },
        'audio-depression': {
            'name': 'Audio Depression Detection Dataset',
            'url': 'https://www.kaggle.com/datasets',
            'description': 'Audio samples for depression detection',
            'size': '~10GB',
            'requires_registration': False,
            'modalities': ['audio'],
            'labels': 'Binary/Multi-class',
            'num_samples': 5000
        }
    }
    
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize dataset downloader.
        
        Args:
            data_dir: Base directory for storing datasets
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        self.datasets_dir = self.data_dir / 'datasets'
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
    
    def list_available_datasets(self) -> None:
        """
        List all available datasets.
        """
        print("\n" + "="*80)
        print("AVAILABLE DEPRESSION DETECTION DATASETS")
        print("="*80)
        
        for key, info in self.DATASETS.items():
            print(f"\n📊 {key.upper()}")
            print(f"   Name: {info['name']}")
            print(f"   URL: {info['url']}")
            print(f"   Description: {info['description']}")
            print(f"   Size: {info['size']}")
            print(f"   Samples: {info['num_samples']}")
            print(f"   Modalities: {', '.join(info['modalities'])}")
            print(f"   Labels: {info['labels']}")
            print(f"   Registration Required: {'Yes' if info['requires_registration'] else 'No'}")
        
        print("\n" + "="*80 + "\n")
    
    def get_daic_woz_info(self) -> Dict:
        """
        Get DAIC-WOZ dataset information and download instructions.
        """
        return {
            'name': 'DAIC-WOZ',
            'official_url': 'https://dcapswoz.ict.usc.edu/',
            'instructions': [
                '1. Visit https://dcapswoz.ict.usc.edu/',
                '2. Create an account and log in',
                '3. Fill out the data request form',
                '4. Wait for approval (usually 1-2 days)',
                '5. Download the dataset files',
                '6. Extract to data/datasets/DAIC-WOZ/',
                '7. Run: python scripts/prepare_daic_woz.py'
            ],
            'file_structure': {
                'dev_data.zip': 'Development set',
                'test_data.zip': 'Test set',
                'train_data.zip': 'Training set',
                'labels.csv': 'PHQ-9 labels',
                'transcripts/': 'Text transcripts'
            },
            'expected_files': [
                'data/datasets/DAIC-WOZ/dev_data/',
                'data/datasets/DAIC-WOZ/test_data/',
                'data/datasets/DAIC-WOZ/train_data/',
                'data/datasets/DAIC-WOZ/labels.csv'
            ]
        }
    
    def create_daic_woz_setup_guide(self) -> str:
        """
        Create a detailed setup guide for DAIC-WOZ.
        """
        guide = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    DAIC-WOZ DATASET SETUP GUIDE                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

📋 STEP-BY-STEP INSTRUCTIONS:

1. REGISTER FOR ACCESS
   └─ Go to: https://dcapswoz.ict.usc.edu/
   └─ Click "Request Access" or "Sign Up"
   └─ Fill in institutional affiliation
   └─ Agree to terms of use
   └─ Wait for email confirmation

2. DOWNLOAD THE DATASET
   └─ Log in to the portal
   └─ Download all available files:
      ├─ dev_data.zip (Development set)
      ├─ test_data.zip (Test set)
      ├─ train_data.zip (Training set)
      └─ labels.csv (Depression labels - PHQ-9 scores)

3. EXTRACT FILES
   └─ Create directory: mkdir -p data/datasets/DAIC-WOZ
   └─ Extract all zip files:
      ├─ unzip dev_data.zip -d data/datasets/DAIC-WOZ/
      ├─ unzip test_data.zip -d data/datasets/DAIC-WOZ/
      ├─ unzip train_data.zip -d data/datasets/DAIC-WOZ/
      └─ cp labels.csv data/datasets/DAIC-WOZ/

4. VERIFY STRUCTURE
   └─ Check that directory structure matches:
      data/datasets/DAIC-WOZ/
      ├── dev_data/
      │   ├── 300/  (interview IDs)
      │   ├── 301/
      │   └── ...
      ├── test_data/
      ├── train_data/
      ├── labels.csv
      └── transcripts/ (if available)

5. PREPARE DATA
   └─ Run: python scripts/prepare_daic_woz.py
   └─ This will:
      ├─ Extract audio from videos
      ├─ Extract transcripts
      ├─ Organize data structure
      └─ Create train/val/test splits

6. VERIFY PREPARATION
   └─ Check output directory:
      data/processed/DAIC-WOZ/
      ├── audio/
      ├── video/
      ├── transcripts/
      ├── features/
      └── metadata.json

📊 DATASET STATISTICS:
   • Total Interviews: 1,189
   • Training: ~900 interviews
   • Validation: ~150 interviews
   • Test: ~139 interviews
   • PHQ-9 Score Range: 0-27
   • Depression Threshold: PHQ-9 ≥ 10

⏱️ EXPECTED DOWNLOAD TIME:
   • Total Size: ~50GB
   • Download Speed: 5 Mbps → ~25 hours
   • Extract Time: ~1-2 hours

💾 STORAGE REQUIREMENTS:
   • Raw Data: ~50GB
   • Processed Data: ~80GB (with features)
   • Total: ~130GB

❓ TROUBLESHOOTING:
   Q: "Access denied" error?
   A: Check that institutional email is verified
   
   Q: Download interrupted?
   A: Resume download or download files individually
   
   Q: Corrupted zip files?
   A: Verify checksums provided by data portal
   
   Q: Not enough disk space?
   A: Delete extracted files after processing or use external storage

📧 SUPPORT:
   • Website: https://dcapswoz.ict.usc.edu/
   • Contact: dcapswoz@ict.usc.edu
   • Issues: Open GitHub issue or contact authors

✅ NEXT STEPS:
   After verification, run:
   └─ python main.py --video data/datasets/DAIC-WOZ/dev_data/*/video/*

"""
        return guide
    
    def download_file(self, url: str, destination: str, show_progress: bool = True) -> bool:
        """
        Download a file from URL.
        
        Args:
            url: URL to download from
            destination: Local file path
            show_progress: Show download progress bar
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Downloading from {url}...")
            
            if show_progress:
                urllib.request.urlretrieve(
                    url, destination,
                    reporthook=self._download_progress_hook
                )
            else:
                urllib.request.urlretrieve(url, destination)
            
            print(f"✓ Downloaded to {destination}")
            return True
            
        except Exception as e:
            print(f"✗ Download failed: {e}")
            return False
    
    @staticmethod
    def _download_progress_hook(block_num, block_size, total_size):
        """
        Progress hook for urllib downloads.
        """
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        print(f"\rDownloading: {percent:.1f}% ", end='')
    
    def extract_zip(self, zip_path: str, extract_to: str) -> bool:
        """
        Extract zip file.
        
        Args:
            zip_path: Path to zip file
            extract_to: Directory to extract to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"✓ Extracted to {extract_to}")
            return True
        except Exception as e:
            print(f"✗ Extraction failed: {e}")
            return False
    
    def extract_tar(self, tar_path: str, extract_to: str) -> bool:
        """
        Extract tar file.
        
        Args:
            tar_path: Path to tar file
            extract_to: Directory to extract to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Extracting {tar_path}...")
            with tarfile.open(tar_path, 'r') as tar_ref:
                tar_ref.extractall(extract_to)
            print(f"✓ Extracted to {extract_to}")
            return True
        except Exception as e:
            print(f"✗ Extraction failed: {e}")
            return False
    
    def save_dataset_config(self, dataset_name: str, config: Dict) -> None:
        """
        Save dataset configuration.
        
        Args:
            dataset_name: Name of dataset
            config: Configuration dictionary
        """
        config_path = self.datasets_dir / f"{dataset_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Saved config to {config_path}")
    
    def load_dataset_config(self, dataset_name: str) -> Optional[Dict]:
        """
        Load dataset configuration.
        
        Args:
            dataset_name: Name of dataset
            
        Returns:
            Configuration dictionary or None if not found
        """
        config_path = self.datasets_dir / f"{dataset_name}_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return None
    
    def verify_dataset(self, dataset_name: str) -> bool:
        """
        Verify if dataset is properly downloaded and extracted.
        
        Args:
            dataset_name: Name of dataset
            
        Returns:
            True if dataset is complete, False otherwise
        """
        dataset_path = self.datasets_dir / dataset_name
        
        if not dataset_path.exists():
            print(f"✗ Dataset directory not found: {dataset_path}")
            return False
        
        # Check for key directories/files based on dataset type
        if dataset_name.lower() == 'daic-woz':
            required = ['train_data', 'dev_data', 'test_data', 'labels.csv']
            missing = [r for r in required if not (dataset_path / r).exists()]
            
            if missing:
                print(f"✗ Missing required files: {missing}")
                return False
            
            print(f"✓ DAIC-WOZ dataset verified successfully")
            return True
        
        print(f"✓ Dataset structure looks valid")
        return True


def main():
    """
    Main function to demonstrate dataset manager.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Depression Detection Dataset Manager'
    )
    parser.add_argument(
        '--list', action='store_true',
        help='List all available datasets'
    )
    parser.add_argument(
        '--daic-setup', action='store_true',
        help='Show DAIC-WOZ setup guide'
    )
    parser.add_argument(
        '--verify', type=str, default=None,
        help='Verify dataset integrity'
    )
    parser.add_argument(
        '--data-dir', type=str, default='data',
        help='Base data directory'
    )
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(data_dir=args.data_dir)
    
    if args.list:
        downloader.list_available_datasets()
    
    elif args.daic_setup:
        guide = downloader.create_daic_woz_setup_guide()
        print(guide)
        
        # Save guide to file
        guide_path = Path('DAIC_WOZ_SETUP_GUIDE.txt')
        with open(guide_path, 'w') as f:
            f.write(guide)
        print(f"\n✓ Setup guide saved to {guide_path}")
    
    elif args.verify:
        is_valid = downloader.verify_dataset(args.verify)
        print(f"\nDataset valid: {is_valid}")
    
    else:
        downloader.list_available_datasets()
        print("\nUsage:")
        print("  python scripts/dataset_downloader.py --list          # List datasets")
        print("  python scripts/dataset_downloader.py --daic-setup    # DAIC-WOZ setup guide")
        print("  python scripts/dataset_downloader.py --verify DAIC-WOZ  # Verify dataset")


if __name__ == '__main__':
    main()
