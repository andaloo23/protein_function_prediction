"""
CNN model for protein function prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CNNModel(nn.Module):
    """
    Convolutional Neural Network model for protein function prediction.
    Uses multiple convolutional layers with different kernel sizes for feature extraction.
    """
    
    def __init__(
        self,
        num_classes: int,
        vocab_size: int,
        embedding_dim: int = 16,
        num_filters: int = 512,
        max_kernel_size: int = 129,
        dense_depth: int = 0,
        max_length: int = 256
    ):
        """
        Initialize the CNN model.
        
        Args:
            num_classes: Number of output classes
            vocab_size: Size of the vocabulary (number of unique amino acids + 1)
            embedding_dim: Dimension of the embedding layer
            num_filters: Number of filters for each convolutional layer
            max_kernel_size: Maximum kernel size for convolutional layers
            dense_depth: Number of dense layers after convolution
            max_length: Maximum sequence length
        """
        super(CNNModel, self).__init__()
        
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.max_kernel_size = max_kernel_size
        self.dense_depth = dense_depth
        self.max_length = max_length
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Convolutional layers with different kernel sizes
        kernel_sizes = list(range(8, max_kernel_size, 8))
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=kernel_size
            )
            for kernel_size in kernel_sizes
        ])
        
        # Max pooling layers for each convolutional layer
        self.pool_layers = nn.ModuleList([
            nn.MaxPool1d(kernel_size=max_length - kernel_size + 1)
            for kernel_size in kernel_sizes
        ])
        
        # Flatten layers
        self.flatten_layers = nn.ModuleList([
            nn.Flatten()
            for _ in range(len(self.conv_layers))
        ])
        
        # Fully connected layers
        input_dim = len(self.conv_layers) * num_filters
        
        if dense_depth > 0:
            self.fc_layers = nn.ModuleList([
                nn.Linear(input_dim if i == 0 else 128, 128)
                for i in range(dense_depth)
            ])
            self.output_layer = nn.Linear(128, num_classes)
        else:
            self.fc_layers = nn.ModuleList()
            self.output_layer = nn.Linear(input_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Embedding: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        x = self.embedding(x)
        
        # Permute for conv1d: (batch_size, embedding_dim, seq_len)
        x = x.permute(0, 2, 1)
        
        # Apply convolution and pooling layers
        conv_outputs = []
        for conv, pool, flatten in zip(self.conv_layers, self.pool_layers, self.flatten_layers):
            conv_out = F.relu(conv(x))
            pooled_out = pool(conv_out)
            flattened_out = flatten(pooled_out)
            conv_outputs.append(flattened_out)
        
        # Concatenate all conv outputs
        x = torch.cat(conv_outputs, dim=1)
        
        # Apply dense layers
        for fc in self.fc_layers:
            x = F.relu(fc(x))
        
        # Output layer
        x = self.output_layer(x)
        
        return x
    
    def get_model_info(self) -> dict:
        """
        Get information about the model architecture.
        
        Returns:
            Dictionary with model information
        """
        kernel_sizes = list(range(8, self.max_kernel_size, 8))
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": "CNN",
            "num_classes": self.num_classes,
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "num_filters": self.num_filters,
            "kernel_sizes": kernel_sizes,
            "dense_depth": self.dense_depth,
            "max_length": self.max_length,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }
