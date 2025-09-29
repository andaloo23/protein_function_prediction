"""
Configuration classes for protein function prediction models.
"""

from dataclasses import dataclass
from typing import List, Optional, Any, Dict
import json
from pathlib import Path


@dataclass
class Config:
    """Base configuration class."""
    
    # Data parameters
    data_path: str = "protein_data.tsv.gz"
    max_sequence_length: int = 256
    min_ec_frequency: int = 100
    test_size: float = 0.2
    random_state: int = 42
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    epochs: int = 100
    early_stop_patience: int = 5
    min_improvement: float = 0.001
    
    # System parameters
    device: str = "cuda"
    num_workers: int = 4
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best: bool = True
    save_every: Optional[int] = None
    
    # Evaluation
    top_k: int = 5
    
    def save(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        config_dict = self.__dict__.copy()
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Configuration saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    def update(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown parameter '{key}' ignored")
    
    def get_model_name(self) -> str:
        """Get model name for saving checkpoints."""
        return "base_model"


@dataclass
class CNNConfig(Config):
    """Configuration for CNN model."""
    
    # CNN-specific parameters
    embedding_dim: int = 16
    num_filters: int = 512
    max_kernel_size: int = 129
    dense_depth: int = 0
    kernel_step: int = 8  # Step size for kernel sizes
    
    def get_model_name(self) -> str:
        """Get model name for saving checkpoints."""
        return f"cnn_model_emb{self.embedding_dim}_filters{self.num_filters}_maxk{self.max_kernel_size}"
    
    def get_kernel_sizes(self) -> List[int]:
        """Get list of kernel sizes for convolutional layers."""
        return list(range(8, self.max_kernel_size, self.kernel_step))


@dataclass
class ESMConfig(Config):
    """Configuration for ESM model."""
    
    # ESM-specific parameters
    esm_model_name: str = "facebook/esm2_t6_8M_UR50D"
    freeze_esm: bool = True
    hidden_dims: List[int] = None
    
    def __post_init__(self):
        """Initialize default hidden dimensions if not provided."""
        if self.hidden_dims is None:
            self.hidden_dims = [320, 200, 128]
    
    def get_model_name(self) -> str:
        """Get model name for saving checkpoints."""
        model_short = self.esm_model_name.split('/')[-1]  # Get just the model name
        freeze_str = "frozen" if self.freeze_esm else "unfrozen"
        return f"esm_model_{model_short}_{freeze_str}"


# Predefined configurations for different scenarios
class ConfigPresets:
    """Predefined configuration presets for common use cases."""
    
    @staticmethod
    def cnn_small() -> CNNConfig:
        """Small CNN configuration for quick experiments."""
        return CNNConfig(
            embedding_dim=8,
            num_filters=128,
            max_kernel_size=65,
            dense_depth=1,
            epochs=50,
            batch_size=32
        )
    
    @staticmethod
    def cnn_medium() -> CNNConfig:
        """Medium CNN configuration (default)."""
        return CNNConfig()
    
    @staticmethod
    def cnn_large() -> CNNConfig:
        """Large CNN configuration for better performance."""
        return CNNConfig(
            embedding_dim=32,
            num_filters=1024,
            max_kernel_size=193,
            dense_depth=2,
            epochs=200,
            batch_size=32,
            learning_rate=0.0005
        )
    
    @staticmethod
    def esm_small() -> ESMConfig:
        """Small ESM configuration for quick experiments."""
        return ESMConfig(
            esm_model_name="facebook/esm2_t6_8M_UR50D",
            hidden_dims=[320, 128],
            epochs=50,
            batch_size=32
        )
    
    @staticmethod
    def esm_medium() -> ESMConfig:
        """Medium ESM configuration (default)."""
        return ESMConfig()
    
    @staticmethod
    def esm_large() -> ESMConfig:
        """Large ESM configuration for better performance."""
        return ESMConfig(
            esm_model_name="facebook/esm2_t12_35M_UR50D",
            hidden_dims=[480, 300, 200, 128],
            freeze_esm=False,
            epochs=100,
            batch_size=16,
            learning_rate=0.0001
        )
    
    @staticmethod
    def debug() -> CNNConfig:
        """Debug configuration for quick testing."""
        return CNNConfig(
            epochs=2,
            batch_size=8,
            early_stop_patience=1,
            save_every=1
        )


def create_config(model_type: str, size: str = "medium", **kwargs) -> Config:
    """
    Create a configuration object.
    
    Args:
        model_type: Type of model ("cnn" or "esm")
        size: Size of configuration ("small", "medium", "large", or "debug")
        **kwargs: Additional parameters to override
    
    Returns:
        Configuration object
    
    Raises:
        ValueError: If model_type or size is invalid
    """
    if model_type == "cnn":
        if size == "small":
            config = ConfigPresets.cnn_small()
        elif size == "medium":
            config = ConfigPresets.cnn_medium()
        elif size == "large":
            config = ConfigPresets.cnn_large()
        elif size == "debug":
            config = ConfigPresets.debug()
        else:
            raise ValueError(f"Unknown size '{size}' for CNN model")
    
    elif model_type == "esm":
        if size == "small":
            config = ConfigPresets.esm_small()
        elif size == "medium":
            config = ConfigPresets.esm_medium()
        elif size == "large":
            config = ConfigPresets.esm_large()
        elif size == "debug":
            debug_config = ConfigPresets.debug()
            config = ESMConfig(**debug_config.__dict__)
        else:
            raise ValueError(f"Unknown size '{size}' for ESM model")
    
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Must be 'cnn' or 'esm'")
    
    # Apply any overrides
    config.update(**kwargs)
    
    return config


def load_config_from_args(args: Any) -> Config:
    """
    Create configuration from command line arguments.
    
    Args:
        args: Parsed command line arguments
    
    Returns:
        Configuration object
    """
    if hasattr(args, 'config_file') and args.config_file:
        # Load from file
        if args.model_type == "cnn":
            config = CNNConfig.load(args.config_file)
        else:
            config = ESMConfig.load(args.config_file)
    else:
        # Create from args
        config = create_config(args.model_type, args.size)
    
    # Override with command line arguments
    override_dict = {}
    for key, value in vars(args).items():
        if value is not None and hasattr(config, key):
            override_dict[key] = value
    
    config.update(**override_dict)
    
    return config
