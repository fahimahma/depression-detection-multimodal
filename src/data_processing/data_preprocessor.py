#!/usr/bin/env python
"""
Dataset preprocessor for depression detection.
Prepares raw data into structured format for training.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.feature_extraction.text_features import TextFeatureExtractor
from src.feature_extraction.audio_features import AudioFeatureExtractor
from src.feature_extraction.facial_features import FacialFeatureExtractor


class DatasetPreprocessor:
    """
    Preprocesses raw dataset into feature-extracted format.
    """
    
    def __init__(self, data_dir: str = 'data', dataset_name: str = 'daic-woz'):
        """
        Initialize preprocessor.
        
        Args:
            data_dir: Base data directory
            dataset_name: Name of dataset
        """
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.raw_dir = self.data_dir / 'datasets' / dataset_name
        self.processed_dir = self.data_dir / 'processed' / dataset_name
        
        # Initialize feature extractors
        self.text_extractor = TextFeatureExtractor()
        self.audio_extractor = AudioFeatureExtractor()
        self.facial_extractor = FacialFeatureExtractor()
        
        # Create output directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        (self.processed_dir / 'audio').mkdir(exist_ok=True)
        (self.processed_dir / 'video').mkdir(exist_ok=True)
        (self.processed_dir / 'transcripts').mkdir(exist_ok=True)
        (self.processed_dir / 'features').mkdir(exist=True)
    
    def process_daic_woz(self) -> None:
        """
        Process DAIC-WOZ dataset.
        """
        print("Processing DAIC-WOZ dataset...")
        
        # Load labels
        labels_path = self.raw_dir / 'labels.csv'
        if not labels_path.exists():
            print(f"✗ Labels not found: {labels_path}")
            return
        
        labels_df = pd.read_csv(labels_path)
        print(f"Loaded {len(labels_df)} labels")
        
        # Process each split
        for split in ['train_data', 'dev_data', 'test_data']:
            split_dir = self.raw_dir / split
            if not split_dir.exists():
                print(f"⚠ Split not found: {split_dir}")
                continue
            
            print(f"\nProcessing {split}...")
            self._process_split(split_dir, split.replace('_data', ''), labels_df)
    
    def _process_split(self, split_dir: Path, split_name: str, labels_df: pd.DataFrame) -> None:
        """
        Process a dataset split.
        
        Args:
            split_dir: Directory containing split data
            split_name: Name of split (train, dev, test)
            labels_df: DataFrame with labels
        """
        interview_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        
        for interview_dir in tqdm(interview_dirs, desc=f"Processing {split_name}"):
            interview_id = interview_dir.name
            
            # Create output directory for this interview
            output_dir = self.processed_dir / interview_id
            output_dir.mkdir(exist_ok=True)
            
            # Extract and save features
            self._process_interview(interview_dir, output_dir, interview_id, labels_df)
    
    def _process_interview(self, interview_dir: Path, output_dir: Path,
                          interview_id: str, labels_df: pd.DataFrame) -> None:
        """
        Process a single interview.
        
        Args:
            interview_dir: Interview directory
            output_dir: Output directory
            interview_id: Interview ID
            labels_df: Labels dataframe
        """
        features = {}
        
        # Get label
        label_row = labels_df[labels_df['interview_id'] == int(interview_id)]
        if not label_row.empty:
            phq9_score = float(label_row.iloc[0]['PHQ8_Score'])
            features['label'] = phq9_score
            features['depressed'] = 1 if phq9_score >= 10 else 0
        
        # Process text/transcript
        transcript_path = self._find_transcript(interview_dir)
        if transcript_path:
            try:
                with open(transcript_path, 'r') as f:
                    transcript_text = f.read()
                text_features = self.text_extractor.extract(transcript_text)
                features['text'] = text_features
                
                # Save transcript
                with open(output_dir / 'transcript.txt', 'w') as f:
                    f.write(transcript_text)
            except Exception as e:
                print(f"  Error processing text for {interview_id}: {e}")
        
        # Process audio
        audio_path = self._find_audio(interview_dir)
        if audio_path:
            try:
                audio_features = self.audio_extractor.extract_from_file(str(audio_path))
                features['audio'] = audio_features
                
                # Copy audio file
                shutil.copy(audio_path, output_dir / 'audio.wav')
            except Exception as e:
                print(f"  Error processing audio for {interview_id}: {e}")
        
        # Process video/facial
        video_path = self._find_video(interview_dir)
        if video_path:
            try:
                facial_features = self.facial_extractor.extract_from_video(str(video_path))
                features['facial'] = facial_features
                
                # Copy video file
                shutil.copy(video_path, output_dir / 'video.mp4')
            except Exception as e:
                print(f"  Error processing video for {interview_id}: {e}")
        
        # Save features as JSON
        features_path = output_dir / 'features.json'
        with open(features_path, 'w') as f:
            json.dump(features, f, indent=2)
    
    @staticmethod
    def _find_transcript(interview_dir: Path) -> Path:
        """
        Find transcript file in interview directory.
        """
        for ext in ['.txt', '.csv', '.json']:
            transcript_files = list(interview_dir.glob(f'*transcript*{ext}'))
            if transcript_files:
                return transcript_files[0]
        return None
    
    @staticmethod
    def _find_audio(interview_dir: Path) -> Path:
        """
        Find audio file in interview directory.
        """
        for ext in ['.wav', '.mp3', '.m4a']:
            audio_files = list(interview_dir.glob(f'*.{ext}'))
            if audio_files:
                return audio_files[0]
        return None
    
    @staticmethod
    def _find_video(interview_dir: Path) -> Path:
        """
        Find video file in interview directory.
        """
        for ext in ['.mp4', '.avi', '.mov']:
            video_files = list(interview_dir.glob(f'*.{ext}'))
            if video_files:
                return video_files[0]
        return None
    
    def create_splits(self, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict:
        """
        Create train/val/test splits.
        
        Args:
            train_ratio: Proportion of training data
            val_ratio: Proportion of validation data
            
        Returns:
            Dictionary with split information
        """
        print("Creating train/val/test splits...")
        
        feature_files = list(self.processed_dir.glob('*/features.json'))
        total_samples = len(feature_files)
        
        # Shuffle and split
        np.random.shuffle(feature_files)
        train_split = int(total_samples * train_ratio)
        val_split = int(total_samples * (train_ratio + val_ratio))
        
        splits = {
            'train': feature_files[:train_split],
            'val': feature_files[train_split:val_split],
            'test': feature_files[val_split:]
        }
        
        # Save split information
        split_info = {
            'train': [str(f.parent.name) for f in splits['train']],
            'val': [str(f.parent.name) for f in splits['val']],
            'test': [str(f.parent.name) for f in splits['test']]
        }
        
        splits_path = self.processed_dir / 'splits.json'
        with open(splits_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"✓ Train: {len(splits['train'])}")
        print(f"✓ Val: {len(splits['val'])}")
        print(f"✓ Test: {len(splits['test'])}")
        print(f"✓ Saved splits to {splits_path}")
        
        return splits
    
    def generate_statistics(self) -> Dict:
        """
        Generate dataset statistics.
        
        Returns:
            Dictionary with statistics
        """
        print("Generating statistics...")
        
        stats = {
            'total_samples': 0,
            'depressed_count': 0,
            'non_depressed_count': 0,
            'phq9_mean': 0.0,
            'phq9_std': 0.0,
            'phq9_min': 0.0,
            'phq9_max': 0.0,
            'modalities': {
                'text': 0,
                'audio': 0,
                'facial': 0
            }
        }
        
        phq9_scores = []
        
        for features_path in self.processed_dir.glob('*/features.json'):
            with open(features_path, 'r') as f:
                features = json.load(f)
            
            stats['total_samples'] += 1
            
            if 'label' in features:
                phq9_scores.append(features['label'])
                if features.get('depressed', 0) == 1:
                    stats['depressed_count'] += 1
                else:
                    stats['non_depressed_count'] += 1
            
            if 'text' in features:
                stats['modalities']['text'] += 1
            if 'audio' in features:
                stats['modalities']['audio'] += 1
            if 'facial' in features:
                stats['modalities']['facial'] += 1
        
        if phq9_scores:
            stats['phq9_mean'] = float(np.mean(phq9_scores))
            stats['phq9_std'] = float(np.std(phq9_scores))
            stats['phq9_min'] = float(np.min(phq9_scores))
            stats['phq9_max'] = float(np.max(phq9_scores))
        
        # Save statistics
        stats_path = self.processed_dir / 'statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✓ Statistics saved to {stats_path}")
        
        return stats


def main():
    """
    Main preprocessing function.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Preprocess depression detection datasets'
    )
    parser.add_argument(
        '--dataset', type=str, default='daic-woz',
        help='Dataset to process'
    )
    parser.add_argument(
        '--data-dir', type=str, default='data',
        help='Base data directory'
    )
    parser.add_argument(
        '--stats-only', action='store_true',
        help='Only generate statistics (skip processing)'
    )
    
    args = parser.parse_args()
    
    preprocessor = DatasetPreprocessor(
        data_dir=args.data_dir,
        dataset_name=args.dataset
    )
    
    if not args.stats_only:
        print(f"Preprocessing {args.dataset} dataset...")
        print(f"Raw data directory: {preprocessor.raw_dir}")
        print(f"Output directory: {preprocessor.processed_dir}")
        
        if args.dataset.lower() == 'daic-woz':
            preprocessor.process_daic_woz()
            preprocessor.create_splits()
    
    # Generate statistics
    stats = preprocessor.generate_statistics()
    print("\nDataset Statistics:")
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
