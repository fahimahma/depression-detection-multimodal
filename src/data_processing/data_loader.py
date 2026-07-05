#!/usr/bin/env python
"""
Dataset loader for depression detection datasets.
Handles loading, preprocessing, and batching of multimodal data.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class DatasetConfig:
    """
    Configuration for dataset loading.
    """
    dataset_name: str = 'daic-woz'
    data_dir: str = 'data'
    split: str = 'train'  # train, val, test
    modalities: List[str] = None
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 4
    sample_rate: int = 16000
    
    def __post_init__(self):
        if self.modalities is None:
            self.modalities = ['text', 'audio', 'facial']


class DepressionDataset(Dataset):
    """
    PyTorch Dataset for depression detection.
    Loads multimodal data (text, audio, facial) and labels.
    """
    
    def __init__(self, data_dir: str, split: str = 'train',
                 modalities: List[str] = None, labels_df: Optional[pd.DataFrame] = None):
        """
        Initialize dataset.
        
        Args:
            data_dir: Root data directory
            split: Data split (train, val, test)
            modalities: List of modalities to load
            labels_df: DataFrame with labels
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.modalities = modalities or ['text', 'audio', 'facial']
        self.labels_df = labels_df
        
        # Load sample IDs
        self.sample_ids = self._load_sample_ids()
    
    def _load_sample_ids(self) -> List[str]:
        """
        Load sample IDs for the split.
        """
        split_dir = self.data_dir / self.split
        if split_dir.exists():
            # Get all subdirectories (sample IDs)
            sample_ids = [d.name for d in split_dir.iterdir() if d.is_dir()]
            return sorted(sample_ids)
        return []
    
    def __len__(self) -> int:
        """
        Get dataset size.
        """
        return len(self.sample_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with modalities and labels
        """
        sample_id = self.sample_ids[idx]
        sample_dir = self.data_dir / self.split / sample_id
        
        sample = {'id': sample_id}
        
        # Load text features
        if 'text' in self.modalities:
            text_path = sample_dir / 'text_features.json'
            if text_path.exists():
                with open(text_path, 'r') as f:
                    sample['text'] = json.load(f)
        
        # Load audio features
        if 'audio' in self.modalities:
            audio_path = sample_dir / 'audio_features.json'
            if audio_path.exists():
                with open(audio_path, 'r') as f:
                    sample['audio'] = json.load(f)
        
        # Load facial features
        if 'facial' in self.modalities:
            facial_path = sample_dir / 'facial_features.json'
            if facial_path.exists():
                with open(facial_path, 'r') as f:
                    sample['facial'] = json.load(f)
        
        # Load label
        if self.labels_df is not None:
            label_row = self.labels_df[self.labels_df['id'] == sample_id]
            if not label_row.empty:
                sample['label'] = float(label_row.iloc[0]['phq9_score'])
                sample['depressed'] = 1 if sample['label'] >= 10 else 0
        
        return sample


class DAICWOZDatasetLoader:
    """
    Loader for DAIC-WOZ dataset.
    """
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize DAIC-WOZ loader.
        
        Args:
            config: Dataset configuration
        """
        self.config = config
        self.data_dir = Path(config.data_dir) / 'datasets' / 'DAIC-WOZ'
        self.labels_df = self._load_labels()
    
    def _load_labels(self) -> pd.DataFrame:
        """
        Load PHQ-9 labels.
        """
        labels_path = self.data_dir / 'labels.csv'
        if labels_path.exists():
            return pd.read_csv(labels_path)
        return pd.DataFrame()
    
    def get_dataset(self) -> DepressionDataset:
        """
        Get PyTorch dataset.
        """
        return DepressionDataset(
            data_dir=self.data_dir,
            split=self.config.split,
            modalities=self.config.modalities,
            labels_df=self.labels_df
        )
    
    def get_dataloader(self) -> DataLoader:
        """
        Get PyTorch dataloader.
        """
        dataset = self.get_dataset()
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers
        )


class DatasetManager:
    """
    Main class for managing all datasets.
    """
    
    SUPPORTED_DATASETS = ['daic-woz', 'd-vlog', 'e-daic']
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize dataset manager.
        
        Args:
            config: Dataset configuration
        """
        self.config = config
        self.data_dir = Path(config.data_dir)
    
    def get_loader(self, split: str = 'train') -> DataLoader:
        """
        Get dataloader for specified split.
        
        Args:
            split: Data split (train, val, test)
            
        Returns:
            PyTorch DataLoader
        """
        self.config.split = split
        
        if self.config.dataset_name.lower() == 'daic-woz':
            loader = DAICWOZDatasetLoader(self.config)
            return loader.get_dataloader()
        else:
            raise ValueError(f"Dataset {self.config.dataset_name} not supported")
    
    def get_train_loader(self) -> DataLoader:
        """
        Get training dataloader.
        """
        return self.get_loader('train')
    
    def get_val_loader(self) -> DataLoader:
        """
        Get validation dataloader.
        """
        self.config.shuffle = False
        return self.get_loader('val')
    
    def get_test_loader(self) -> DataLoader:
        """
        Get test dataloader.
        """
        self.config.shuffle = False
        return self.get_loader('test')
    
    def get_dataset_info(self) -> Dict:
        """
        Get dataset information.
        """
        return {
            'name': self.config.dataset_name,
            'modalities': self.config.modalities,
            'split': self.config.split,
            'batch_size': self.config.batch_size,
            'data_dir': str(self.data_dir)
        }


def demo_usage():
    """
    Demonstrate dataset loading.
    """
    # Create config
    config = DatasetConfig(
        dataset_name='daic-woz',
        data_dir='data',
        split='train',
        modalities=['text', 'audio', 'facial'],
        batch_size=16
    )
    
    # Create manager
    manager = DatasetManager(config)
    
    print("Dataset Info:")
    print(json.dumps(manager.get_dataset_info(), indent=2))
    
    # Try to get loader
    try:
        train_loader = manager.get_train_loader()
        print(f"✓ Got training dataloader")
        
        # Iterate through a few batches
        for i, batch in enumerate(train_loader):
            if i >= 2:  # Just show 2 batches
                break
            print(f"\nBatch {i+1}:")
            print(f"  IDs: {batch['id']}")
            if 'label' in batch:
                print(f"  Labels: {batch['label']}")
    
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("Make sure dataset is downloaded and prepared.")


if __name__ == '__main__':
    demo_usage()
