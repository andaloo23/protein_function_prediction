"""
ESM model for protein function prediction.
"""

import torch
import torch.nn as nn
from transformers import EsmModel
from typing import Tuple


class ESMModel(nn.Module):
    """
    ESM (Evolutionary Scale Modeling) based model for protein function prediction.
    Uses pre-trained ESM embeddings with a classification head.
    """
    
    def __init__(
        self,
        num_classes: int,
        esm_model_name: str = "facebook/esm2_t6_8M_UR50D",
        freeze_esm: bool = True,
        hidden_dims: list = [320, 200, 128]
    ):
        """
        Initialize the ESM model.
        
        Args:
            num_classes: Number of output classes
            esm_model_name: Name of the pre-trained ESM model
            freeze_esm: Whether to freeze ESM parameters
            hidden_dims: Dimensions of the classification head layers
        """
        super(ESMModel, self).__init__()
        
        self.num_classes = num_classes
        self.esm_model_name = esm_model_name
        self.freeze_esm = freeze_esm
        self.hidden_dims = hidden_dims
        
        # Load pre-trained ESM model
        self.esm_model = EsmModel.from_pretrained(esm_model_name)
        
        # Freeze ESM parameters if specified
        if freeze_esm:
            for param in self.esm_model.parameters():
                param.requires_grad = False
        
        # Classification head
        layers = []
        input_dim = hidden_dims[0]  # ESM embedding dimension
        
        for i, hidden_dim in enumerate(hidden_dims[1:], 1):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])
            input_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(input_dim, num_classes))
        
        self.fc_layers = nn.Sequential(*layers)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length)
            attention_mask: Attention mask of shape (batch_size, sequence_length)
            
        Returns:
            Tuple of (logits, probabilities)
        """
        # Get ESM embeddings
        outputs = self.esm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use mean pooling on the last hidden state
        # Shape: (batch_size, seq_len, hidden_dim) -> (batch_size, hidden_dim)
        pooled_output = torch.mean(outputs.hidden_states[-1], dim=1)
        
        # Apply classification head
        logits = self.fc_layers(pooled_output)
        
        # Compute probabilities
        probabilities = torch.softmax(logits, dim=1)
        
        return logits, probabilities
    
    def get_model_info(self) -> dict:
        """
        Get information about the model architecture.
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            "model_type": "ESM",
            "esm_model_name": self.esm_model_name,
            "num_classes": self.num_classes,
            "freeze_esm": self.freeze_esm,
            "hidden_dims": self.hidden_dims,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": frozen_params,
        }
