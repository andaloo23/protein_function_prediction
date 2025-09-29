"""
Training module for protein function prediction models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
from typing import Dict, List, Tuple, Optional, Any
import os
from pathlib import Path


class Trainer:
    """
    Generic trainer class for protein function prediction models.
    Handles training loop, validation, early stopping, and checkpointing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        device: str = "cuda",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
        checkpoint_dir: str = "checkpoints",
        model_name: str = "model"
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The neural network model
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            device: Device to train on ("cuda" or "cpu")
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            class_weights: Weights for class balancing
            checkpoint_dir: Directory to save checkpoints
            model_name: Name prefix for saved models
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.top5_accuracies = []
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.best_train_loss = float('inf')
        self.no_improvement_count = 0
    
    def train_epoch(self, model_type: str = "cnn") -> float:
        """
        Train for one epoch.
        
        Args:
            model_type: Type of model ("cnn" or "esm")
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm.tqdm(self.train_dataloader, desc="Training"):
            if model_type == "cnn":
                inputs, labels = batch
                inputs = torch.tensor(inputs).to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(inputs)
                
            elif model_type == "esm":
                tokenized_inputs, labels = batch
                input_ids = torch.tensor(tokenized_inputs["input_ids"]).to(self.device)
                attention_mask = torch.tensor(tokenized_inputs["attention_mask"]).to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits, _ = self.model(input_ids, attention_mask)
            
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            
            # Compute loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, model_type: str = "cnn") -> Tuple[float, float, float]:
        """
        Validate the model.
        
        Args:
            model_type: Type of model ("cnn" or "esm")
            
        Returns:
            Tuple of (validation_loss, top1_accuracy, top5_accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total_samples = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm.tqdm(self.val_dataloader, desc="Validation"):
                if model_type == "cnn":
                    inputs, labels = batch
                    inputs = torch.tensor(inputs).to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass
                    logits = self.model(inputs)
                    probs = torch.softmax(logits, dim=1)
                    
                elif model_type == "esm":
                    tokenized_inputs, labels = batch
                    input_ids = torch.tensor(tokenized_inputs["input_ids"]).to(self.device)
                    attention_mask = torch.tensor(tokenized_inputs["attention_mask"]).to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass
                    logits, probs = self.model(input_ids, attention_mask)
                
                else:
                    raise ValueError(f"Unknown model_type: {model_type}")
                
                # Compute loss
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                num_batches += 1
                
                # Get predictions
                pred_top1 = probs.argmax(dim=1)
                pred_top5 = probs.topk(5, dim=1)[1]
                
                # Convert labels to class indices
                true_labels = labels.argmax(dim=1)
                
                # Calculate accuracies
                correct_top1 += (pred_top1 == true_labels).sum().item()
                
                for i in range(pred_top5.shape[0]):
                    if true_labels[i] in pred_top5[i]:
                        correct_top5 += 1
                
                total_samples += labels.shape[0]
        
        avg_loss = total_loss / num_batches
        top1_accuracy = correct_top1 / total_samples
        top5_accuracy = correct_top5 / total_samples
        
        return avg_loss, top1_accuracy, top5_accuracy
    
    def train(
        self,
        epochs: int = 100,
        model_type: str = "cnn",
        early_stop_patience: int = 5,
        min_improvement: float = 0.001,
        save_best: bool = True,
        save_every: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            epochs: Number of epochs to train
            model_type: Type of model ("cnn" or "esm")
            early_stop_patience: Number of epochs to wait before early stopping
            min_improvement: Minimum improvement required to reset patience
            save_best: Whether to save the best model
            save_every: Save checkpoint every N epochs (if specified)
            
        Returns:
            Dictionary with training history
        """
        print(f"Starting training for {epochs} epochs...")
        print(f"Model type: {model_type}")
        print(f"Device: {self.device}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train for one epoch
            train_loss = self.train_epoch(model_type)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, top1_acc, top5_acc = self.validate(model_type)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(top1_acc)
            self.top5_accuracies.append(top5_acc)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Top-1 Accuracy: {top1_acc:.4f}")
            print(f"Top-5 Accuracy: {top5_acc:.4f}")
            
            # Check for improvement
            if self.best_val_loss - val_loss > min_improvement:
                self.best_val_loss = val_loss
                self.no_improvement_count = 0
                
                if save_best:
                    self.save_checkpoint(epoch, is_best=True)
            else:
                self.no_improvement_count += 1
            
            # Save checkpoint periodically
            if save_every and (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch)
            
            # Early stopping
            if self.no_improvement_count >= early_stop_patience:
                print(f"\nEarly stopping after {epoch + 1} epochs (no improvement for {early_stop_patience} epochs)")
                break
        
        print("\\nTraining completed!")
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "top5_accuracies": self.top5_accuracies
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "top5_accuracies": self.top5_accuracies,
            "best_val_loss": self.best_val_loss,
        }
        
        if is_best:
            filename = f"{self.model_name}_best.pth"
        else:
            filename = f"{self.model_name}_epoch_{epoch + 1}.pth"
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint: {filepath}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Epoch number from checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Restore training history
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.val_accuracies = checkpoint.get("val_accuracies", [])
        self.top5_accuracies = checkpoint.get("top5_accuracies", [])
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        
        epoch = checkpoint["epoch"]
        print(f"Loaded checkpoint from epoch {epoch}")
        
        return epoch
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary of training progress.
        
        Returns:
            Dictionary with training summary
        """
        if not self.train_losses:
            return {"message": "No training history available"}
        
        return {
            "epochs_trained": len(self.train_losses),
            "best_val_loss": self.best_val_loss,
            "best_val_accuracy": max(self.val_accuracies) if self.val_accuracies else 0.0,
            "best_top5_accuracy": max(self.top5_accuracies) if self.top5_accuracies else 0.0,
            "final_train_loss": self.train_losses[-1],
            "final_val_loss": self.val_losses[-1],
            "final_val_accuracy": self.val_accuracies[-1] if self.val_accuracies else 0.0,
        }
