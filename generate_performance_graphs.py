"""
Generate performance visualization graphs for NAS models.
This script creates comprehensive visualizations comparing different NAS methods.
"""
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import os
import glob
from tqdm import tqdm
import time

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Configuration
CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
NUM_CLASSES = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import model architectures
from backend.model import NASModel
from backend.github_model import load_github_checkpoint

# ==================== PREPROCESSING ====================

# Random Search NAS preprocessing
preprocess_random = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Gradient-based NAS preprocessing (EfficientNet-B4)
preprocess_gradient = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== MODEL LOADING ====================

def load_random_nas_model():
    """Load Random Search NAS model"""
    architecture = {
        'conv_layers': [
            {'filters': 128, 'kernel_size': 2, 'activation': 'relu'},
            {'filters': 32, 'kernel_size': 2, 'activation': 'relu'},
            {'filters': 32, 'kernel_size': 2, 'activation': 'relu'},
            {'filters': 16, 'kernel_size': 2, 'activation': 'leakyrelu'},
            {'filters': 32, 'kernel_size': 2, 'activation': 'leakyrelu'},
            {'filters': 128, 'kernel_size': 5, 'activation': 'leakyrelu'}
        ],
        'fc_layers': [
            {'units': 128, 'activation': 'leakyrelu'},
            {'units': 128, 'activation': 'relu'}
        ],
        'dropout_rate': 0.3
    }
    
    model = NASModel(architecture, NUM_CLASSES)
    model_path = os.path.join("backend", "models", "the_nas_model.pth")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def load_gradient_nas_model():
    """Load Gradient-based NAS model (EfficientNet-B4)"""
    model_path = "best.pth"
    model = load_github_checkpoint(model_path, device=device)
    return model

# ==================== DATA LOADING ====================

def load_test_data(test_dir="Testing"):
    """Load test images and labels"""
    images = []
    labels = []
    image_paths = []
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_path = os.path.join(test_dir, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: {class_path} not found")
            continue
        
        # Get all image files
        img_files = glob.glob(os.path.join(class_path, "*.jpg")) + \
                   glob.glob(os.path.join(class_path, "*.jpeg")) + \
                   glob.glob(os.path.join(class_path, "*.png"))
        
        for img_path in img_files[:50]:  # Limit to 50 images per class for speed
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                labels.append(class_idx)
                image_paths.append(img_path)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    return images, labels, image_paths

# ==================== PREDICTION ====================

def predict_batch(model, images, preprocess):
    """Get predictions for a batch of images"""
    predictions = []
    probabilities = []
    inference_times = []
    
    for img in tqdm(images, desc="Predicting"):
        # Preprocess
        input_tensor = preprocess(img).unsqueeze(0).to(device)
        
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        pred = torch.argmax(probs).item()
        predictions.append(pred)
        probabilities.append(probs.cpu().numpy())
        inference_times.append(inference_time)
    
    return np.array(predictions), np.array(probabilities), np.array(inference_times)

# ==================== VISUALIZATION FUNCTIONS ====================

def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")

def plot_accuracy_comparison(accuracies, save_path):
    """Plot accuracy comparison bar chart"""
    models = list(accuracies.keys())
    accs = list(accuracies.values())
    
    plt.figure(figsize=(10, 6))
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = plt.bar(models, accs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accs):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.xlabel('NAS Method', fontsize=12, fontweight='bold')
    plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")

def plot_class_wise_performance(y_true, predictions_dict, save_path):
    """Plot class-wise precision, recall, F1-score"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics_names = ['Precision', 'Recall', 'F1-Score']
    
    for idx, metric_name in enumerate(metrics_names):
        ax = axes[idx]
        
        # Calculate metrics for each model
        x = np.arange(len(CLASS_NAMES))
        width = 0.25
        
        for i, (model_name, preds) in enumerate(predictions_dict.items()):
            from sklearn.metrics import precision_recall_fscore_support
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, preds, average=None, labels=range(NUM_CLASSES), zero_division=0
            )
            
            if metric_name == 'Precision':
                values = precision
            elif metric_name == 'Recall':
                values = recall
            else:
                values = f1
            
            offset = width * (i - 1)
            bars = ax.bar(x + offset, values, width, label=model_name, alpha=0.8)
        
        ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Tumor Class', fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_name} by Class', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([name.replace('_', '\n') for name in CLASS_NAMES], fontsize=9)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")

def plot_inference_time_comparison(times_dict, save_path):
    """Plot inference time comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box plot
    data = [times for times in times_dict.values()]
    labels = list(times_dict.keys())
    
    bp = ax1.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
    for patch, color in zip(bp['boxes'], ['#3498db', '#e74c3c', '#2ecc71']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('Inference Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('NAS Method', fontsize=12, fontweight='bold')
    ax1.set_title('Inference Time Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Bar plot with average times
    avg_times = [np.mean(times) for times in times_dict.values()]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax2.bar(labels, avg_times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar, time_val in zip(bars, avg_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.2f} ms',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.set_ylabel('Average Inference Time (ms)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('NAS Method', fontsize=12, fontweight='bold')
    ax2.set_title('Average Inference Time', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")

def plot_confidence_distribution(probs_dict, y_true, predictions_dict, save_path):
    """Plot confidence distribution for correct vs incorrect predictions"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (model_name, probs) in enumerate(probs_dict.items()):
        ax = axes[idx]
        preds = predictions_dict[model_name]
        
        # Get max confidence for each prediction
        confidences = np.max(probs, axis=1)
        
        # Separate correct and incorrect predictions
        correct_mask = (preds == y_true)
        correct_conf = confidences[correct_mask]
        incorrect_conf = confidences[~correct_mask]
        
        # Plot histograms
        ax.hist(correct_conf, bins=20, alpha=0.7, label='Correct', color='green', edgecolor='black')
        ax.hist(incorrect_conf, bins=20, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
        
        ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name}\nConfidence Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")

def plot_roc_curves(y_true, probs_dict, save_path):
    """Plot ROC curves for all models"""
    # Binarize labels for multi-class ROC
    y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.ravel()
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for class_idx in range(NUM_CLASSES):
        ax = axes[class_idx]
        
        for model_idx, (model_name, probs) in enumerate(probs_dict.items()):
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true_bin[:, class_idx], probs[:, class_idx])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=colors[model_idx], lw=2,
                   label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
        ax.set_title(f'ROC Curve - {CLASS_NAMES[class_idx].replace("_", " ").title()}',
                    fontsize=12, fontweight='bold')
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")

def plot_model_comparison_table(results_dict, save_path):
    """Create a comprehensive comparison table"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    models = list(results_dict.keys())
    metrics = ['Accuracy (%)', 'Precision', 'Recall', 'F1-Score', 'Avg Inference (ms)']
    
    table_data = []
    for model in models:
        row = [
            model,
            f"{results_dict[model]['accuracy']:.2f}",
            f"{results_dict[model]['precision']:.3f}",
            f"{results_dict[model]['recall']:.3f}",
            f"{results_dict[model]['f1']:.3f}",
            f"{results_dict[model]['inference_time']:.2f}"
        ]
        table_data.append(row)
    
    table = ax.table(cellText=table_data, 
                    colLabels=['Model'] + metrics,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(metrics) + 1):
        cell = table[(0, i)]
        cell.set_facecolor('#3498db')
        cell.set_text_props(weight='bold', color='white')
    
    # Style rows
    colors_alt = ['#ecf0f1', '#ffffff']
    for i in range(1, len(models) + 1):
        for j in range(len(metrics) + 1):
            cell = table[(i, j)]
            cell.set_facecolor(colors_alt[i % 2])
    
    plt.title('Comprehensive Model Performance Comparison', 
             fontsize=16, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")

# ==================== MAIN EXECUTION ====================

def main():
    print("=" * 60)
    print("🎨 GENERATING PERFORMANCE VISUALIZATION GRAPHS")
    print("=" * 60)
    
    # Create output directory
    output_dir = "performance_graphs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data
    print("\n📁 Loading test data...")
    images, y_true, image_paths = load_test_data()
    
    if len(images) == 0:
        print("❌ No test images found! Please ensure Testing/ directory exists with images.")
        print("   Expected structure: Testing/glioma_tumor/, Testing/meningioma_tumor/, etc.")
        return
    
    y_true = np.array(y_true)
    print(f"✓ Loaded {len(images)} test images")
    print(f"  Class distribution: {np.bincount(y_true)}")
    
    # Dictionary to store results
    predictions_dict = {}
    probabilities_dict = {}
    inference_times_dict = {}
    results_summary = {}
    
    # Model configurations
    models_config = [
        ("Random Search NAS", load_random_nas_model, preprocess_random),
        ("Gradient-Based NAS", load_gradient_nas_model, preprocess_gradient),
    ]
    
    # Evaluate each model
    for model_name, load_func, preprocess in models_config:
        print(f"\n🔄 Evaluating {model_name}...")
        
        try:
            model = load_func()
            preds, probs, times = predict_batch(model, images, preprocess)
            
            predictions_dict[model_name] = preds
            probabilities_dict[model_name] = probs
            inference_times_dict[model_name] = times
            
            # Calculate metrics
            accuracy = np.mean(preds == y_true) * 100
            from sklearn.metrics import precision_recall_fscore_support
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, preds, average='weighted', zero_division=0
            )
            
            results_summary[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'inference_time': np.mean(times)
            }
            
            print(f"  ✓ Accuracy: {accuracy:.2f}%")
            print(f"  ✓ Avg Inference Time: {np.mean(times):.2f} ms")
            
        except Exception as e:
            print(f"  ❌ Error evaluating {model_name}: {e}")
            continue
    
    if len(predictions_dict) == 0:
        print("\n❌ No models could be evaluated!")
        return
    
    # Generate visualizations
    print("\n📊 Generating graphs...")
    
    # 1. Confusion matrices
    for model_name, preds in predictions_dict.items():
        safe_name = model_name.replace(" ", "_").lower()
        plot_confusion_matrix(y_true, preds, model_name,
                            f"{output_dir}/01_confusion_matrix_{safe_name}.png")
    
    # 2. Accuracy comparison
    accuracies = {name: results['accuracy'] for name, results in results_summary.items()}
    plot_accuracy_comparison(accuracies, f"{output_dir}/02_accuracy_comparison.png")
    
    # 3. Class-wise performance
    plot_class_wise_performance(y_true, predictions_dict,
                               f"{output_dir}/03_classwise_performance.png")
    
    # 4. Inference time comparison
    plot_inference_time_comparison(inference_times_dict,
                                   f"{output_dir}/04_inference_time_comparison.png")
    
    # 5. Confidence distribution
    plot_confidence_distribution(probabilities_dict, y_true, predictions_dict,
                                f"{output_dir}/05_confidence_distribution.png")
    
    # 6. ROC curves
    plot_roc_curves(y_true, probabilities_dict,
                   f"{output_dir}/06_roc_curves_multiclass.png")
    
    # 7. Comparison table
    plot_model_comparison_table(results_summary,
                               f"{output_dir}/07_comprehensive_comparison.png")
    
    print("\n" + "=" * 60)
    print("✅ ALL GRAPHS GENERATED SUCCESSFULLY!")
    print(f"📂 Saved in: {output_dir}/")
    print("=" * 60)
    
    # Print summary
    print("\n📋 RESULTS SUMMARY:")
    print("-" * 60)
    for model_name, results in results_summary.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy:       {results['accuracy']:.2f}%")
        print(f"  Precision:      {results['precision']:.3f}")
        print(f"  Recall:         {results['recall']:.3f}")
        print(f"  F1-Score:       {results['f1']:.3f}")
        print(f"  Inference Time: {results['inference_time']:.2f} ms")
    print("-" * 60)

if __name__ == "__main__":
    main()
