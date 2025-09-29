"""
Evaluation module for protein function prediction models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import tqdm
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path


class Evaluator:
    """
    Comprehensive evaluation class for protein function prediction models.
    Handles detailed metrics, visualizations, and analysis.
    """
    
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: str = "cuda",
        index_to_ec: Optional[Dict[int, str]] = None
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained neural network model
            dataloader: Data loader for evaluation
            device: Device to run evaluation on
            index_to_ec: Mapping from class indices to EC numbers
        """
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.index_to_ec = index_to_ec or {}
        
        # Results storage
        self.predictions = []
        self.true_labels = []
        self.probabilities = []
        
        # Metrics storage
        self.class_metrics = {}
        self.overall_metrics = {}
    
    def evaluate(self, model_type: str = "cnn", top_k: int = 5) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of the model.
        
        Args:
            model_type: Type of model ("cnn" or "esm")
            top_k: Number of top predictions to consider
            
        Returns:
            Dictionary with evaluation results
        """
        print("Starting evaluation...")
        
        self.model.eval()
        
        # Initialize counters
        correct_top1 = 0
        correct_topk = 0
        total_samples = 0
        
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        class_false_positives = defaultdict(int)
        class_false_negatives = defaultdict(int)
        
        all_predictions = []
        all_true_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm.tqdm(self.dataloader, desc="Evaluating"):
                if model_type == "cnn":
                    inputs, labels = batch
                    inputs = torch.tensor(inputs).to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass
                    logits = self.model(inputs)
                    probabilities = torch.softmax(logits, dim=1)
                    
                elif model_type == "esm":
                    tokenized_inputs, labels = batch
                    input_ids = torch.tensor(tokenized_inputs["input_ids"]).to(self.device)
                    attention_mask = torch.tensor(tokenized_inputs["attention_mask"]).to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass
                    logits, probabilities = self.model(input_ids, attention_mask)
                
                else:
                    raise ValueError(f"Unknown model_type: {model_type}")
                
                # Get predictions
                pred_top1 = probabilities.argmax(dim=1)
                pred_topk = probabilities.topk(top_k, dim=1)[1]
                
                # Convert labels to class indices
                true_labels = labels.argmax(dim=1)
                
                # Store results
                all_predictions.extend(pred_top1.cpu().numpy())
                all_true_labels.extend(true_labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Calculate accuracies
                correct_top1 += (pred_top1 == true_labels).sum().item()
                
                for i in range(pred_topk.shape[0]):
                    if true_labels[i] in pred_topk[i]:
                        correct_topk += 1
                
                total_samples += labels.shape[0]
                
                # Per-class metrics
                for i in range(labels.shape[0]):
                    true_class = true_labels[i].item()
                    pred_class = pred_top1[i].item()
                    
                    class_total[true_class] += 1
                    
                    if pred_class == true_class:
                        class_correct[true_class] += 1
                    else:
                        class_false_positives[pred_class] += 1
                        class_false_negatives[true_class] += 1
        
        # Store results
        self.predictions = all_predictions
        self.true_labels = all_true_labels
        self.probabilities = all_probabilities
        
        # Calculate overall metrics
        top1_accuracy = correct_top1 / total_samples
        topk_accuracy = correct_topk / total_samples
        
        self.overall_metrics = {
            "top1_accuracy": top1_accuracy,
            f"top{top_k}_accuracy": topk_accuracy,
            "total_samples": total_samples,
        }
        
        # Calculate per-class metrics
        self._calculate_class_metrics(class_correct, class_total, class_false_positives, class_false_negatives)
        
        # Calculate additional metrics
        macro_f1, weighted_f1 = self._calculate_f1_scores()
        self.overall_metrics["macro_f1"] = macro_f1
        self.overall_metrics["weighted_f1"] = weighted_f1
        
        print(f"Evaluation completed!")
        print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
        print(f"Top-{top_k} Accuracy: {topk_accuracy:.4f}")
        print(f"Macro F1-score: {macro_f1:.4f}")
        print(f"Weighted F1-score: {weighted_f1:.4f}")
        
        return {
            "overall_metrics": self.overall_metrics,
            "class_metrics": self.class_metrics,
        }
    
    def _calculate_class_metrics(
        self,
        class_correct: Dict[int, int],
        class_total: Dict[int, int],
        class_false_positives: Dict[int, int],
        class_false_negatives: Dict[int, int]
    ) -> None:
        """Calculate per-class precision, recall, F1-score, and accuracy."""
        
        for class_idx in class_total.keys():
            tp = class_correct[class_idx]
            fp = class_false_positives[class_idx]
            fn = class_false_negatives[class_idx]
            total = class_total[class_idx]
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = tp / total if total > 0 else 0.0
            
            ec_number = self.index_to_ec.get(class_idx, f"Class_{class_idx}")
            
            self.class_metrics[class_idx] = {
                "ec_number": ec_number,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "total_samples": total,
            }
    
    def _calculate_f1_scores(self) -> Tuple[float, float]:
        """Calculate macro and weighted F1-scores."""
        f1_scores = []
        weighted_f1_sum = 0
        total_samples = 0
        
        for metrics in self.class_metrics.values():
            f1_scores.append(metrics["f1_score"])
            weighted_f1_sum += metrics["f1_score"] * metrics["total_samples"]
            total_samples += metrics["total_samples"]
        
        macro_f1 = np.mean(f1_scores) if f1_scores else 0.0
        weighted_f1 = weighted_f1_sum / total_samples if total_samples > 0 else 0.0
        
        return macro_f1, weighted_f1
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Generate confusion matrix.
        
        Returns:
            Confusion matrix as numpy array
        """
        if not self.predictions or not self.true_labels:
            raise ValueError("No evaluation results available. Run evaluate() first.")
        
        num_classes = len(self.class_metrics)
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        
        for pred, true in zip(self.predictions, self.true_labels):
            confusion_matrix[true, pred] += 1
        
        return confusion_matrix
    
    def plot_accuracy_distribution(self, save_path: Optional[str] = None) -> None:
        """
        Plot distribution of per-class accuracies.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.class_metrics:
            raise ValueError("No evaluation results available. Run evaluate() first.")
        
        accuracies = [metrics["accuracy"] for metrics in self.class_metrics.values()]
        
        plt.figure(figsize=(10, 6))
        plt.boxplot(accuracies)
        plt.ylabel("Accuracy")
        plt.title("Distribution of Per-Class Accuracies")
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        q1 = np.percentile(accuracies, 25)
        median = np.percentile(accuracies, 50)
        q3 = np.percentile(accuracies, 75)
        
        plt.text(0.02, 0.98, f"Median: {median:.3f}\\nQ1: {q1:.3f}\\nQ3: {q3:.3f}", 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_class_accuracies(self, save_path: Optional[str] = None) -> None:
        """
        Plot accuracies for each class.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.class_metrics:
            raise ValueError("No evaluation results available. Run evaluate() first.")
        
        # Sort by accuracy for better visualization
        sorted_classes = sorted(self.class_metrics.items(), key=lambda x: x[1]["accuracy"])
        
        accuracies = [metrics["accuracy"] for _, metrics in sorted_classes]
        ec_numbers = [metrics["ec_number"] for _, metrics in sorted_classes]
        
        plt.figure(figsize=(15, 8))
        bars = plt.bar(range(len(accuracies)), accuracies, color='green', alpha=0.7)
        plt.xlabel("EC Number")
        plt.ylabel("Accuracy")
        plt.title("Per-Class Accuracy")
        plt.xticks(range(len(ec_numbers)), ec_numbers, rotation=90)
        plt.grid(True, alpha=0.3)
        
        # Add average line
        avg_accuracy = np.mean(accuracies)
        plt.axhline(y=avg_accuracy, color='red', linestyle='--', 
                   label=f'Average: {avg_accuracy:.3f}')
        plt.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def identify_outliers(self, threshold: float = 1.5) -> List[Dict[str, Any]]:
        """
        Identify outlier classes based on accuracy.
        
        Args:
            threshold: IQR threshold for outlier detection
            
        Returns:
            List of outlier class information
        """
        if not self.class_metrics:
            raise ValueError("No evaluation results available. Run evaluate() first.")
        
        accuracies = [metrics["accuracy"] for metrics in self.class_metrics.values()]
        
        q1 = np.percentile(accuracies, 25)
        q3 = np.percentile(accuracies, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        outliers = []
        for class_idx, metrics in self.class_metrics.items():
            accuracy = metrics["accuracy"]
            if accuracy < lower_bound or accuracy > upper_bound:
                outlier_info = {
                    "class_index": class_idx,
                    "ec_number": metrics["ec_number"],
                    "accuracy": accuracy,
                    "total_samples": metrics["total_samples"],
                    "outlier_type": "low" if accuracy < lower_bound else "high"
                }
                outliers.append(outlier_info)
        
        return outliers
    
    def get_top_errors(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get classes with the most prediction errors.
        
        Args:
            top_n: Number of top error classes to return
            
        Returns:
            List of classes with most errors
        """
        if not self.class_metrics:
            raise ValueError("No evaluation results available. Run evaluate() first.")
        
        error_classes = []
        for class_idx, metrics in self.class_metrics.items():
            total_errors = metrics["false_positives"] + metrics["false_negatives"]
            error_rate = total_errors / (metrics["total_samples"] + metrics["false_positives"])
            
            error_classes.append({
                "class_index": class_idx,
                "ec_number": metrics["ec_number"],
                "total_errors": total_errors,
                "error_rate": error_rate,
                "false_positives": metrics["false_positives"],
                "false_negatives": metrics["false_negatives"],
                "accuracy": metrics["accuracy"]
            })
        
        # Sort by total errors (descending)
        error_classes.sort(key=lambda x: x["total_errors"], reverse=True)
        
        return error_classes[:top_n]
    
    def print_detailed_results(self, show_all_classes: bool = False) -> None:
        """
        Print detailed evaluation results.
        
        Args:
            show_all_classes: Whether to show metrics for all classes
        """
        if not self.overall_metrics or not self.class_metrics:
            raise ValueError("No evaluation results available. Run evaluate() first.")
        
        print("\\n" + "="*50)
        print("DETAILED EVALUATION RESULTS")
        print("="*50)
        
        # Overall metrics
        print("\\nOVERALL METRICS:")
        print("-" * 20)
        for metric, value in self.overall_metrics.items():
            if isinstance(value, float):
                print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
            else:
                print(f"{metric.replace('_', ' ').title()}: {value}")
        
        # Class-wise metrics
        if show_all_classes:
            print("\\nPER-CLASS METRICS:")
            print("-" * 20)
            print(f"{'EC Number':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Samples':<10}")
            print("-" * 70)
            
            for class_idx in sorted(self.class_metrics.keys()):
                metrics = self.class_metrics[class_idx]
                print(f"{metrics['ec_number']:<12} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
                      f"{metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f} {metrics['total_samples']:<10}")
        
        # Outliers
        outliers = self.identify_outliers()
        if outliers:
            print(f"\\nOUTLIER CLASSES ({len(outliers)} found):")
            print("-" * 30)
            for outlier in outliers:
                print(f"  {outlier['ec_number']}: {outlier['accuracy']:.4f} accuracy "
                      f"({outlier['total_samples']} samples) - {outlier['outlier_type']} performer")
        
        # Top errors
        top_errors = self.get_top_errors(5)
        print(f"\\nTOP ERROR CLASSES:")
        print("-" * 20)
        for error_class in top_errors:
            print(f"  {error_class['ec_number']}: {error_class['total_errors']} errors "
                  f"(FP: {error_class['false_positives']}, FN: {error_class['false_negatives']})")
    
    def save_results(self, output_dir: str) -> None:
        """
        Save evaluation results to files.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON-like format
        import json
        
        results = {
            "overall_metrics": self.overall_metrics,
            "class_metrics": {str(k): v for k, v in self.class_metrics.items()},
        }
        
        with open(output_path / "evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Save plots
        self.plot_accuracy_distribution(save_path=output_path / "accuracy_distribution.png")
        self.plot_class_accuracies(save_path=output_path / "class_accuracies.png")
        
        print(f"Results saved to {output_path}")
