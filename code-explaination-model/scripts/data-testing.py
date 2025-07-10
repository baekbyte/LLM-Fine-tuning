"""
Testing and evaluation script for processed data
"""

import json
import torch
from pathlib import Path
from transformers import GPT2Tokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def test_tokenized_data():
    """Test that tokenized data loads correctly"""
    print("Testing tokenized data...")
    
    tokenized_files = [
        "data/tokenized/train_tokenized.pt",
        "data/tokenized/validation_tokenized.pt",
        "data/tokenized/test_tokenized.pt"
    ]
    
    for file_path in tokenized_files:
        if Path(file_path).exists():
            try:
                data = torch.load(file_path)
                print(f"{file_path}: {data['input_ids'].shape}")
                
                # Test a sample
                sample_ids = data['input_ids'][0]
                tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                tokenizer.pad_token = tokenizer.eos_token
                
                # Add special tokens
                special_tokens = ["<|code|>", "<|explanation|>"]
                tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
                
                decoded = tokenizer.decode(sample_ids, skip_special_tokens=False)
                print(f"Sample: {decoded[:100]}...")
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        else:
            print(f"{file_path} not found")

def visualize_data_distribution():
    """Create visualizations of the data distribution"""
    print("\nCreating data distribution visualizations...")
    
    # Load quality reports
    reports = {}
    for split in ['train', 'validation', 'test']:
        report_path = f"reports/{split}_quality_report.json"
        if Path(report_path).exists():
            with open(report_path, 'r') as f:
                reports[split] = json.load(f)
    
    if not reports:
        print("No quality reports found")
        return
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Data Quality Analysis', fontsize=16)
    
    # Plot 1: Valid examples percentage
    splits = list(reports.keys())
    valid_percentages = [reports[split]['structure']['valid_percentage'] for split in splits]
    
    axes[0, 0].bar(splits, valid_percentages, color='skyblue')
    axes[0, 0].set_title('Valid Examples Percentage')
    axes[0, 0].set_ylabel('Percentage')
    axes[0, 0].set_ylim(0, 100)
    
    # Plot 2: Code quality metrics
    syntactically_valid = [reports[split]['code']['syntactically_valid'] for split in splits]
    has_functions = [reports[split]['code']['has_functions'] for split in splits]
    
    x = np.arange(len(splits))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, syntactically_valid, width, label='Syntactically Valid', color='lightgreen')
    axes[0, 1].bar(x + width/2, has_functions, width, label='Has Functions', color='lightcoral')
    axes[0, 1].set_title('Code Quality Metrics')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(splits)
    axes[0, 1].legend()
    
    # Plot 3: Length distributions
    avg_code_lengths = [reports[split]['code']['avg_code_length'] for split in splits]
    avg_explanation_lengths = [reports[split]['explanation']['avg_explanation_length'] for split in splits]
    
    axes[1, 0].bar(x - width/2, avg_code_lengths, width, label='Avg Code Length (lines)', color='lightblue')
    axes[1, 1].bar(x + width/2, avg_explanation_lengths, width, label='Avg Explanation Length (words)', color='lightyellow')
    axes[1, 0].set_title('Average Code Length')
    axes[1, 0].set_ylabel('Lines')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(splits)
    
    axes[1, 1].set_title('Average Explanation Length')
    axes[1, 1].set_ylabel('Words')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(splits)
    
    plt.tight_layout()
    plt.savefig('reports/data_quality_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved to reports/data_quality_visualization.png")

def check_data_consistency():
    """Check consistency across train/validation/test splits"""
    print("\nChecking data consistency across splits...")
    
    # Load cleaned data
    splits_data = {}
    for split in ['train', 'validation', 'test']:
        file_path = f"data/cleaned/{split}_cleaned.json"
        if Path(file_path).exists():
            with open(file_path, 'r') as f:
                splits_data[split] = json.load(f)
    
    if not splits_data:
        print("No cleaned data found")
        return
    
    # Check for data leakage (duplicate examples across splits)
    print("Checking for data leakage...")
    
    all_codes = {}
    for split_name, data in splits_data.items():
        for i, example in enumerate(data):
            code = example['code']
            if code in all_codes:
                print(f"Duplicate code found in {split_name} (index {i}) and {all_codes[code]}")
            else:
                all_codes[code] = f"{split_name} (index {i})"
    
    print(f"Checked {len(all_codes)} unique code examples")
    
    # Check split sizes
    print("\nSplit sizes:")
    for split_name, data in splits_data.items():
        print(f"{split_name}: {len(data)} examples")
    
    # Check data distribution
    total_examples = sum(len(data) for data in splits_data.values())
    print(f"\nTotal examples: {total_examples}")
    
    for split_name, data in splits_data.items():
        percentage = len(data) / total_examples * 100
        print(f"{split_name}: {percentage:.1f}%")

if __name__ == "__main__":
    print("Running data testing and evaluation...")
    
    test_tokenized_data()
    visualize_data_distribution()
    check_data_consistency()
    
    print("\nTesting complete!")