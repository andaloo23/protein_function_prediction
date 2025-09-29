"""
Data processing module for protein function prediction.
Handles data loading, preprocessing, tokenization, and dataset creation.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from transformers import AutoTokenizer
from typing import Tuple, List, Dict, Any, Optional


class ProteinDataset(Dataset):
    """Custom Dataset for protein sequences and labels."""
    
    def __init__(self, sequences: List[str], labels: np.ndarray, tokenizer_func=None):
        """
        Initialize the dataset.
        
        Args:
            sequences: List of protein sequences
            labels: Encoded labels
            tokenizer_func: Function to tokenize sequences (for CNN model)
        """
        self.sequences = sequences
        self.labels = labels
        self.tokenizer_func = tokenizer_func
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[Any, torch.Tensor]:
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        if self.tokenizer_func:
            # For CNN model - apply custom tokenization
            sequence = self.tokenizer_func(sequence)
        
        return sequence, label


class DataProcessor:
    """Handles all data processing operations for protein function prediction."""
    
    def __init__(self, data_path: str = "protein_data.tsv.gz", max_length: int = 256):
        """
        Initialize the data processor.
        
        Args:
            data_path: Path to the protein data file
            max_length: Maximum sequence length to keep
        """
        self.data_path = data_path
        self.max_length = max_length
        self.data = None
        self.encoder = None
        self.vocab = None
        self.tokens = None
        self.vocab_size = None
        self.num_classes = None
        self.ec_to_index = {}
        self.index_to_ec = {}
        
    def load_and_preprocess_data(self) -> pd.DataFrame:
        """
        Load and preprocess the protein data.
        
        Returns:
            Preprocessed DataFrame
        """
        print("Loading protein data...")
        self.data = pd.read_csv(self.data_path, sep="\t")
        
        # Filter by sequence length
        initial_count = len(self.data)
        self.data = self.data[self.data["Length"] <= self.max_length]
        print(f"Filtered sequences: {initial_count} -> {len(self.data)} (max length: {self.max_length})")
        
        # Reduce EC number to 3 digits
        self.data = self.data.dropna(subset=['EC number']).assign(
            ec_number_reduced=lambda x: x['EC number'].str.split(".").str[:3].str.join(".")
        )
        
        # Filter out EC numbers with frequency less than 100
        ec_number_counts = self.data["ec_number_reduced"].value_counts()
        valid_ec_numbers = ec_number_counts[ec_number_counts >= 100].index
        self.data = self.data[self.data["ec_number_reduced"].isin(valid_ec_numbers)]
        
        print(f"Final dataset size: {len(self.data)} sequences")
        print(f"Number of EC classes: {len(valid_ec_numbers)}")
        
        return self.data
    
    def build_vocabulary(self) -> None:
        """Build vocabulary from protein sequences (for CNN model)."""
        if self.data is None:
            raise ValueError("Data must be loaded first. Call load_and_preprocess_data().")
        
        print("Building vocabulary...")
        vocab = set()
        for sequence in self.data["Sequence"]:
            for amino_acid in sequence:
                vocab.add(amino_acid)
        
        self.tokens = "".join(sorted(list(vocab)))
        self.vocab_size = len(self.tokens) + 1  # +1 for padding token
        self.vocab = vocab
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Amino acids: {self.tokens}")
    
    def setup_labels(self) -> None:
        """Setup label encoding."""
        if self.data is None:
            raise ValueError("Data must be loaded first. Call load_and_preprocess_data().")
        
        print("Setting up label encoding...")
        self.encoder = OneHotEncoder(sparse_output=False)
        self.encoded_labels = self.encoder.fit_transform(self.data[["ec_number_reduced"]])
        self.num_classes = len(self.encoder.categories_[0])
        
        # Create mappings
        categories = self.encoder.categories_[0]
        self.ec_to_index = {category: index for index, category in enumerate(categories)}
        self.index_to_ec = {index: category for index, category in enumerate(categories)}
        
        print(f"Number of classes: {self.num_classes}")
    
    def tokenize_sequence(self, sequence: str, max_length: Optional[int] = None) -> np.ndarray:
        """
        Tokenize a protein sequence for CNN model.
        
        Args:
            sequence: Protein sequence string
            max_length: Maximum length (defaults to self.max_length)
            
        Returns:
            Tokenized sequence as numpy array
        """
        if self.tokens is None:
            raise ValueError("Vocabulary must be built first. Call build_vocabulary().")
        
        if max_length is None:
            max_length = self.max_length
            
        aa_to_idx = {aa: idx + 1 for idx, aa in enumerate(self.tokens)}
        encoding = np.zeros(max_length, dtype=int)
        
        for i, aa in enumerate(sequence):
            if i >= max_length:
                break
            if aa in aa_to_idx:
                encoding[i] = aa_to_idx[aa]
        
        return encoding
    
    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Split data into training and validation sets.
        
        Args:
            test_size: Fraction of data to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, val_data, train_labels, val_labels)
        """
        if self.data is None or self.encoded_labels is None:
            raise ValueError("Data and labels must be setup first.")
        
        print(f"Splitting data (train: {1-test_size:.1%}, val: {test_size:.1%})...")
        
        # Shuffle data
        data_shuffled, labels_shuffled = shuffle(
            self.data, self.encoded_labels, random_state=random_state
        )
        
        # Split
        train_size = int((1 - test_size) * len(data_shuffled))
        train_data = data_shuffled[:train_size]
        val_data = data_shuffled[train_size:]
        train_labels = labels_shuffled[:train_size]
        val_labels = labels_shuffled[train_size:]
        
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        
        return train_data, val_data, train_labels, val_labels
    
    def create_cnn_datasets(self, train_data: pd.DataFrame, val_data: pd.DataFrame, 
                           train_labels: np.ndarray, val_labels: np.ndarray) -> Tuple[ProteinDataset, ProteinDataset]:
        """
        Create datasets for CNN model.
        
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        train_dataset = ProteinDataset(
            train_data["Sequence"].tolist(),
            train_labels,
            tokenizer_func=self.tokenize_sequence
        )
        
        val_dataset = ProteinDataset(
            val_data["Sequence"].tolist(),
            val_labels,
            tokenizer_func=self.tokenize_sequence
        )
        
        return train_dataset, val_dataset
    
    def create_esm_datasets(self, train_data: pd.DataFrame, val_data: pd.DataFrame,
                           train_labels: np.ndarray, val_labels: np.ndarray) -> Tuple[ProteinDataset, ProteinDataset]:
        """
        Create datasets for ESM model.
        
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        train_dataset = ProteinDataset(
            train_data["Sequence"].tolist(),
            train_labels
        )
        
        val_dataset = ProteinDataset(
            val_data["Sequence"].tolist(),
            val_labels
        )
        
        return train_dataset, val_dataset
    
    def create_dataloaders(self, train_dataset: ProteinDataset, val_dataset: ProteinDataset,
                          batch_size: int = 64, model_type: str = "cnn") -> Tuple[DataLoader, DataLoader]:
        """
        Create DataLoaders for training and validation.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            batch_size: Batch size
            model_type: Type of model ("cnn" or "esm")
            
        Returns:
            Tuple of (train_dataloader, val_dataloader)
        """
        if model_type == "cnn":
            def collate_fn(batch):
                sequences = [item[0] for item in batch]
                labels = [item[1] for item in batch]
                return sequences, torch.tensor(labels)
        
        elif model_type == "esm":
            tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
            
            def collate_fn(batch):
                sequences = [item[0] for item in batch]
                labels = [item[1] for item in batch]
                return tokenizer(sequences, padding=True), torch.tensor(labels)
        
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False
        )
        
        return train_dataloader, val_dataloader
    
    def get_class_weights(self, device: str = "cuda") -> torch.Tensor:
        """
        Calculate class weights for balanced training.
        
        Args:
            device: Device to put the weights tensor on
            
        Returns:
            Class weights tensor
        """
        if self.data is None or self.ec_to_index is None:
            raise ValueError("Data and labels must be setup first.")
        
        # Count occurrences of each class
        class_counts = [0] * self.num_classes
        for ec in self.data["ec_number_reduced"]:
            index = self.ec_to_index[ec]
            class_counts[index] += 1
        
        # Calculate inverse frequency weights
        class_counts = torch.tensor(class_counts, dtype=torch.float32)
        class_weights = class_counts.min() / class_counts
        
        return class_weights.to(device)
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about the processed data.
        
        Returns:
            Dictionary with data information
        """
        if self.data is None:
            return {"error": "Data not loaded"}
        
        return {
            "total_samples": len(self.data),
            "num_classes": self.num_classes,
            "vocab_size": self.vocab_size,
            "max_length": self.max_length,
            "ec_classes": list(self.ec_to_index.keys()),
            "avg_sequence_length": self.data["Sequence"].str.len().mean(),
        }
