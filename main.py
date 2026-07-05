#!/usr/bin/env python
"""
Main script for depression detection using multimodal fusion.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

from src.feature_extraction.text_features import TextFeatureExtractor
from src.feature_extraction.audio_features import AudioFeatureExtractor
from src.feature_extraction.facial_features import FacialFeatureExtractor


class DepressionDetector:
    """
    Main class for depression detection using multimodal fusion.
    """
    
    def __init__(self, text_weight: float = 1.0, audio_weight: float = 1.0,
                 facial_weight: float = 1.0):
        """
        Initialize the depression detector.
        
        Args:
            text_weight: Weight for text features
            audio_weight: Weight for audio features
            facial_weight: Weight for facial features
        """
        self.text_extractor = TextFeatureExtractor()
        self.audio_extractor = AudioFeatureExtractor()
        self.facial_extractor = FacialFeatureExtractor()
        
        self.text_weight = text_weight
        self.audio_weight = audio_weight
        self.facial_weight = facial_weight
    
    def detect_from_video(self, video_path: str, transcript_path: str = None) -> Dict[str, Any]:
        """
        Detect depression from video file.
        
        Args:
            video_path: Path to video file
            transcript_path: Path to transcript file (optional)
            
        Returns:
            Dictionary with detection results
        """
        results = {
            'video_path': video_path,
            'status': 'processing',
            'features': {},
            'prediction': None,
            'risk_level': None,
            'confidence': None
        }
        
        try:
            # Extract text features
            if transcript_path and os.path.exists(transcript_path):
                with open(transcript_path, 'r') as f:
                    transcript = f.read()
                text_features = self.text_extractor.extract(transcript)
                results['features']['text'] = text_features
            else:
                text_features = {}
                print(f"Warning: Transcript not found at {transcript_path}")
            
            # Extract audio features (from video)
            print(f"Extracting audio features from {video_path}...")
            audio_features = self.audio_extractor.extract_from_file(video_path)
            results['features']['audio'] = audio_features
            
            # Extract facial features (from video)
            print(f"Extracting facial features from {video_path}...")
            facial_features = self.facial_extractor.extract_from_video(video_path)
            results['features']['facial'] = facial_features
            
            # Compute depression score (simplified)
            depression_score = self._compute_depression_score(
                text_features, audio_features, facial_features
            )
            
            # Determine risk level
            risk_level = self._determine_risk_level(depression_score)
            
            # Compute confidence
            confidence = self._compute_confidence(
                text_features, audio_features, facial_features
            )
            
            results['status'] = 'success'
            results['prediction'] = float(depression_score)
            results['risk_level'] = risk_level
            results['confidence'] = float(confidence)
            
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
            print(f"Error during depression detection: {e}")
        
        return results
    
    def _compute_depression_score(self, text_features: Dict, audio_features: Dict,
                                 facial_features: Dict) -> float:
        """
        Compute depression score from multimodal features.
        
        Args:
            text_features: Text feature dictionary
            audio_features: Audio feature dictionary
            facial_features: Facial feature dictionary
            
        Returns:
            Depression score (0-1)
        """
        scores = []
        
        # Text-based score
        if text_features:
            text_score = self._compute_text_score(text_features)
            scores.append((text_score, self.text_weight))
        
        # Audio-based score
        if audio_features:
            audio_score = self._compute_audio_score(audio_features)
            scores.append((audio_score, self.audio_weight))
        
        # Facial-based score
        if facial_features:
            facial_score = self._compute_facial_score(facial_features)
            scores.append((facial_score, self.facial_weight))
        
        if not scores:
            return 0.5
        
        # Weighted average
        weighted_score = sum(score * weight for score, weight in scores)
        total_weight = sum(weight for _, weight in scores)
        
        depression_score = weighted_score / total_weight if total_weight > 0 else 0.5
        
        return np.clip(depression_score, 0.0, 1.0)
    
    def _compute_text_score(self, text_features: Dict) -> float:
        """Compute depression score from text features."""
        score = 0.0
        
        # Sentiment analysis
        if 'sent_polarity' in text_features:
            polarity = text_features['sent_polarity']
            score += (1.0 - (polarity + 1) / 2) * 0.3  # Negative sentiment
        
        # Depression indicators
        dep_indicators = [v for k, v in text_features.items() if k.startswith('dep_ind_')]
        if dep_indicators:
            score += min(sum(dep_indicators) / 10, 1.0) * 0.4
        
        # Psycholinguistic features
        psyc_negative = text_features.get('psyc_negative_words', 0)
        psyc_positive = text_features.get('psyc_positive_words', 1)
        score += (psyc_negative / (psyc_positive + psyc_negative + 1e-10)) * 0.3
        
        return np.clip(score, 0.0, 1.0)
    
    def _compute_audio_score(self, audio_features: Dict) -> float:
        """Compute depression score from audio features."""
        score = 0.0
        
        # Pitch variation (low pitch variation indicates depression)
        if 'f0_std' in audio_features:
            f0_std = audio_features['f0_std']
            score += (1.0 - min(f0_std / 50, 1.0)) * 0.3  # Low pitch variation
        
        # Energy (low energy indicates depression)
        if 'energy_mean' in audio_features:
            energy = audio_features['energy_mean']
            score += (1.0 - min(energy / 0.1, 1.0)) * 0.3  # Low energy
        
        # Speech rate (changes indicate depression)
        if 'speech_rate' in audio_features:
            speech_rate = audio_features['speech_rate']
            score += (1.0 - min(speech_rate / 0.5, 1.0)) * 0.2  # Slow speech
        
        # Jitter and shimmer (voice quality)
        if 'jitter' in audio_features and audio_features['jitter'] > 0:
            jitter = audio_features['jitter']
            score += min(jitter * 5, 1.0) * 0.2  # Voice instability
        
        return np.clip(score, 0.0, 1.0)
    
    def _compute_facial_score(self, facial_features: Dict) -> float:
        """Compute depression score from facial features."""
        score = 0.0
        
        # Eye aspect ratio (lower = less eye movement)
        if 'landmark_eye_aspect_avg_mean' in facial_features:
            ear = facial_features['landmark_eye_aspect_avg_mean']
            score += (1.0 - min(ear, 1.0)) * 0.3  # Low eye openness
        
        # Mouth openness (lower = less expression)
        if 'landmark_mouth_openness_mean' in facial_features:
            mouth = facial_features['landmark_mouth_openness_mean']
            score += (1.0 - min(mouth / 0.1, 1.0)) * 0.3
        
        # Expression brightness (darkness indicates depression)
        if 'expression_brightness_mean_mean' in facial_features:
            brightness = facial_features['expression_brightness_mean_mean']
            score += (1.0 - brightness / 255) * 0.2  # Dark face
        
        # Skin texture (stress shows in texture)
        if 'skin_roughness_mean' in facial_features:
            roughness = facial_features['skin_roughness_mean']
            score += min(roughness * 10, 1.0) * 0.2
        
        return np.clip(score, 0.0, 1.0)
    
    def _determine_risk_level(self, score: float) -> str:
        """
        Determine risk level from depression score.
        
        Args:
            score: Depression score (0-1)
            
        Returns:
            Risk level string
        """
        if score < 0.3:
            return 'Low'
        elif score < 0.6:
            return 'Moderate'
        elif score < 0.8:
            return 'High'
        else:
            return 'Critical'
    
    def _compute_confidence(self, text_features: Dict, audio_features: Dict,
                           facial_features: Dict) -> float:
        """
        Compute confidence score based on feature completeness.
        
        Args:
            text_features: Text feature dictionary
            audio_features: Audio feature dictionary
            facial_features: Facial feature dictionary
            
        Returns:
            Confidence score (0-1)
        """
        completeness = 0.0
        count = 0
        
        if text_features:
            completeness += len(text_features) / 50  # Normalize
            count += 1
        
        if audio_features:
            completeness += len(audio_features) / 100  # Normalize
            count += 1
        
        if facial_features:
            completeness += len(facial_features) / 100  # Normalize
            count += 1
        
        if count == 0:
            return 0.0
        
        confidence = completeness / count
        return np.clip(confidence, 0.0, 1.0)


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(
        description='Depression Detection using Multimodal Fusion'
    )
    
    parser.add_argument('--video', type=str, required=True,
                       help='Path to video file')
    parser.add_argument('--transcript', type=str, default=None,
                       help='Path to transcript file')
    parser.add_argument('--output', type=str, default='results.json',
                       help='Output file for results')
    parser.add_argument('--text-weight', type=float, default=1.0,
                       help='Weight for text features')
    parser.add_argument('--audio-weight', type=float, default=1.0,
                       help='Weight for audio features')
    parser.add_argument('--facial-weight', type=float, default=1.0,
                       help='Weight for facial features')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = DepressionDetector(
        text_weight=args.text_weight,
        audio_weight=args.audio_weight,
        facial_weight=args.facial_weight
    )
    
    # Detect depression
    results = detector.detect_from_video(
        video_path=args.video,
        transcript_path=args.transcript
    )
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDepression Detection Results:")
    print(f"Status: {results['status']}")
    if results['status'] == 'success':
        print(f"Depression Score: {results['prediction']:.3f}")
        print(f"Risk Level: {results['risk_level']}")
        print(f"Confidence: {results['confidence']:.3f}")
    
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
