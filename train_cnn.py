"""
Main training script for CNN model.
"""

import argparse
import torch
from pathlib import Path

from src.data import DataProcessor
from src.models import CNNModel
from src.training import Trainer
from src.evaluation import Evaluator
from src.config import create_config, CNNConfig


def main():
    parser = argparse.ArgumentParser(description="Train CNN model for protein function prediction")
    
    # Model configuration
    parser.add_argument("--size", type=str, default="medium", 
                       choices=["small", "medium", "large", "debug"],
                       help="Model size configuration")
    parser.add_argument("--config-file", type=str, help="Path to configuration file")
    
    # Data parameters
    parser.add_argument("--data-path", type=str, default="protein_data.tsv.gz",
                       help="Path to protein data file")
    parser.add_argument("--max-length", type=int, default=256,
                       help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for training")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                       help="Weight decay")
    parser.add_argument("--early-stop-patience", type=int, default=5,
                       help="Early stopping patience")
    
    # Model parameters
    parser.add_argument("--embedding-dim", type=int, default=16,
                       help="Embedding dimension")
    parser.add_argument("--num-filters", type=int, default=512,
                       help="Number of filters per convolutional layer")
    parser.add_argument("--max-kernel-size", type=int, default=129,
                       help="Maximum kernel size")
    parser.add_argument("--dense-depth", type=int, default=0,
                       help="Number of dense layers after convolution")
    
    # System parameters
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for training")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                       help="Directory to save checkpoints")
    
    # Evaluation parameters
    parser.add_argument("--evaluate", action="store_true",
                       help="Run evaluation after training")
    parser.add_argument("--checkpoint-path", type=str,
                       help="Path to checkpoint for evaluation only")
    parser.add_argument("--eval-only", action="store_true",
                       help="Only run evaluation (requires --checkpoint-path)")
    
    args = parser.parse_args()
    
    # Set model type for config creation
    args.model_type = "cnn"
    
    # Create configuration
    if args.config_file:
        config = CNNConfig.load(args.config_file)
        # Override with command line arguments
        for key, value in vars(args).items():
            if value is not None and hasattr(config, key):
                setattr(config, key, value)
    else:
        config = create_config("cnn", args.size, **{k: v for k, v in vars(args).items() if v is not None})
    
    print("Configuration:")
    print("-" * 40)
    for key, value in config.__dict__.items():
        print(f"{key}: {value}")
    print("-" * 40)
    
    # Set device
    device = config.device
    if not torch.cuda.is_available() and device == "cuda":
        print("CUDA not available, using CPU")
        device = "cpu"
        config.device = device
    
    print(f"Using device: {device}")
    
    # Initialize data processor
    print("\nInitializing data processor...")
    data_processor = DataProcessor(
        data_path=config.data_path,
        max_length=config.max_sequence_length
    )
    
    # Load and preprocess data
    data = data_processor.load_and_preprocess_data()
    data_processor.build_vocabulary()
    data_processor.setup_labels()
    
    print(f"Data info: {data_processor.get_data_info()}")
    
    # Split data
    train_data, val_data, train_labels, val_labels = data_processor.split_data(
        test_size=config.test_size,
        random_state=config.random_state
    )
    
    # Create datasets and dataloaders
    train_dataset, val_dataset = data_processor.create_cnn_datasets(
        train_data, val_data, train_labels, val_labels
    )
    
    train_dataloader, val_dataloader = data_processor.create_dataloaders(
        train_dataset, val_dataset,
        batch_size=config.batch_size,
        model_type="cnn"
    )
    
    # Initialize model
    print("\nInitializing CNN model...")
    model = CNNModel(
        num_classes=data_processor.num_classes,
        vocab_size=data_processor.vocab_size,
        embedding_dim=config.embedding_dim,
        num_filters=config.num_filters,
        max_kernel_size=config.max_kernel_size,
        dense_depth=config.dense_depth,
        max_length=config.max_sequence_length
    )
    
    print(f"Model info: {model.get_model_info()}")
    
    # Get class weights for balanced training
    class_weights = data_processor.get_class_weights(device)
    
    if args.eval_only:
        # Only run evaluation
        if not args.checkpoint_path:
            raise ValueError("--checkpoint-path required for evaluation only mode")
        
        print(f"\nLoading model from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Run evaluation
        evaluator = Evaluator(
            model=model,
            dataloader=val_dataloader,
            device=device,
            index_to_ec=data_processor.index_to_ec
        )
        
        results = evaluator.evaluate(model_type="cnn", top_k=config.top_k)
        evaluator.print_detailed_results(show_all_classes=False)
        
        # Save evaluation results
        output_dir = Path(config.checkpoint_dir) / "evaluation_results"
        evaluator.save_results(str(output_dir))
        
        return
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        class_weights=class_weights,
        checkpoint_dir=config.checkpoint_dir,
        model_name=config.get_model_name()
    )
    
    # Load checkpoint if specified
    start_epoch = 0
    if args.checkpoint_path:
        print(f"Loading checkpoint from {args.checkpoint_path}")
        start_epoch = trainer.load_checkpoint(args.checkpoint_path)
    
    # Train model
    print("\nStarting training...")
    training_history = trainer.train(
        epochs=config.epochs - start_epoch,
        model_type="cnn",
        early_stop_patience=config.early_stop_patience,
        min_improvement=config.min_improvement,
        save_best=config.save_best,
        save_every=config.save_every
    )
    
    # Print training summary
    print("\nTraining Summary:")
    summary = trainer.get_training_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Save final configuration
    config_path = Path(config.checkpoint_dir) / f"{config.get_model_name()}_config.json"
    config.save(str(config_path))
    
    # Run evaluation if requested
    if args.evaluate:
        print("\nRunning final evaluation...")
        evaluator = Evaluator(
            model=model,
            dataloader=val_dataloader,
            device=device,
            index_to_ec=data_processor.index_to_ec
        )
        
        results = evaluator.evaluate(model_type="cnn", top_k=config.top_k)
        evaluator.print_detailed_results(show_all_classes=False)
        
        # Save evaluation results
        output_dir = Path(config.checkpoint_dir) / "evaluation_results"
        evaluator.save_results(str(output_dir))
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
