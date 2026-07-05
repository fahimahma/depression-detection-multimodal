import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

class EarlyFusion(nn.Module):
    """
    Early Fusion Strategy: Concatenate all features and process through shared network.
    """
    
    def __init__(self, text_dim: int, audio_dim: int, facial_dim: int, 
                 hidden_dim: int = 512, output_dim: int = 2):
        super().__init__()
        
        total_dim = text_dim + audio_dim + facial_dim
        
        self.fusion_network = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, text_features: torch.Tensor, audio_features: torch.Tensor,
                facial_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_features: (batch_size, text_dim)
            audio_features: (batch_size, audio_dim)
            facial_features: (batch_size, facial_dim)
            
        Returns:
            output: (batch_size, output_dim)
        """
        # Concatenate features
        fused = torch.cat([text_features, audio_features, facial_features], dim=1)
        
        # Process through network
        output = self.fusion_network(fused)
        
        return output


class MidFusion(nn.Module):
    """
    Mid Fusion Strategy: Process modalities separately, then fuse at intermediate layer.
    """
    
    def __init__(self, text_dim: int, audio_dim: int, facial_dim: int,
                 hidden_dim: int = 512, fusion_dim: int = 256, output_dim: int = 2):
        super().__init__()
        
        # Modality-specific processing
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, fusion_dim)
        )
        
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, fusion_dim)
        )
        
        self.facial_encoder = nn.Sequential(
            nn.Linear(facial_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, fusion_dim)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=1)
        )
        
        # Classification network
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, text_features: torch.Tensor, audio_features: torch.Tensor,
                facial_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_features: (batch_size, text_dim)
            audio_features: (batch_size, audio_dim)
            facial_features: (batch_size, facial_dim)
            
        Returns:
            output: (batch_size, output_dim)
        """
        # Encode each modality
        text_encoded = self.text_encoder(text_features)
        audio_encoded = self.audio_encoder(audio_features)
        facial_encoded = self.facial_encoder(facial_features)
        
        # Concatenate encoded features
        fused = torch.cat([text_encoded, audio_encoded, facial_encoded], dim=1)
        
        # Apply attention
        attention_weights = self.attention(fused)
        
        # Weighted combination
        weighted_fused = fused * attention_weights.repeat(1, 3)
        
        # Classify
        output = self.classifier(weighted_fused)
        
        return output


class LateFusion(nn.Module):
    """
    Late Fusion Strategy: Train separate classifiers for each modality, then combine.
    """
    
    def __init__(self, text_dim: int, audio_dim: int, facial_dim: int,
                 hidden_dim: int = 512, output_dim: int = 2):
        super().__init__()
        
        # Text classifier
        self.text_classifier = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Audio classifier
        self.audio_classifier = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Facial classifier
        self.facial_classifier = nn.Sequential(
            nn.Linear(facial_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Fusion weights
        self.fusion_weights = nn.Parameter(
            torch.tensor([1.0, 1.0, 1.0]), requires_grad=True
        )
    
    def forward(self, text_features: torch.Tensor, audio_features: torch.Tensor,
                facial_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_features: (batch_size, text_dim)
            audio_features: (batch_size, audio_dim)
            facial_features: (batch_size, facial_dim)
            
        Returns:
            output: (batch_size, output_dim)
        """
        # Get predictions from each modality
        text_pred = self.text_classifier(text_features)
        audio_pred = self.audio_classifier(audio_features)
        facial_pred = self.facial_classifier(facial_features)
        
        # Normalize weights
        weights = F.softmax(self.fusion_weights, dim=0)
        
        # Weighted combination
        output = (
            weights[0] * text_pred +
            weights[1] * audio_pred +
            weights[2] * facial_pred
        )
        
        return output


class TransformerFusion(nn.Module):
    """
    Transformer-based Fusion: Use attention mechanism across modalities.
    """
    
    def __init__(self, text_dim: int, audio_dim: int, facial_dim: int,
                 hidden_dim: int = 256, num_heads: int = 4, 
                 num_layers: int = 2, output_dim: int = 2):
        super().__init__()
        
        # Project all modalities to same dimension
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.audio_projection = nn.Linear(audio_dim, hidden_dim)
        self.facial_projection = nn.Linear(facial_dim, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, text_features: torch.Tensor, audio_features: torch.Tensor,
                facial_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_features: (batch_size, text_dim)
            audio_features: (batch_size, audio_dim)
            facial_features: (batch_size, facial_dim)
            
        Returns:
            output: (batch_size, output_dim)
        """
        # Project features
        text_proj = self.text_projection(text_features)
        audio_proj = self.audio_projection(audio_features)
        facial_proj = self.facial_projection(facial_features)
        
        # Stack into sequence (batch_size, 3, hidden_dim)
        stacked = torch.stack([text_proj, audio_proj, facial_proj], dim=1)
        
        # Apply transformer
        transformed = self.transformer(stacked)
        
        # Flatten and classify
        flattened = transformed.reshape(transformed.shape[0], -1)
        output = self.classifier(flattened)
        
        return output
