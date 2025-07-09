"""
Collect CodeT5 training datasets for code-to-explanation fine-tuning from Hugging Face
"""

import os
import json
import pandas as pd
from datasets import load_dataset
from pathlib import Path
import requests
import zipfile
import gzip

def setup_directories():
    """
    Create necessary directories for data storage
    """

    directories = ['data/raw', 'data/processed', 'data/splits']
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("Created directory structure")

def download_codexglue_dataset():
    """
    Download CodeXGLUE code summarization dataset
    """

    print("Downloading CodeXGLUE Code Summarization dataset...")
    
    # CodeXGLUE has datasets on Hugging Face
    try:
        # Load Python code summarization dataset
        dataset = load_dataset("code_x_glue_ct_code_to_text", "python")
        
        # Save raw data
        dataset.save_to_disk("data/raw/codexglue_python")
        
        print(f"Downloaded dataset with {len(dataset['train'])} training examples")
        return dataset
        
    except Exception as e:
        print(f"Error downloading CodeXGLUE dataset: {e}")
        return None


def explore_dataset_structure(dataset, name):
    """
    Explore the structure of a dataset to understand its format
    """
    print(f"\nExploring {name} dataset structure:")
    print(f"  Dataset keys: {list(dataset.keys())}")
    
    if 'train' in dataset:
        train_data = dataset['train']
        print(f"Training samples: {len(train_data)}")
        print(f"Feature columns: {list(train_data.features.keys())}")
        
        # Show a sample
        if len(train_data) > 0:
            sample = train_data[0]
            print(f"Sample data structure:")
            for key, value in sample.items():
                if isinstance(value, str):
                    preview = value[:100] + "..." if len(value) > 100 else value
                    print(f"{key}: {preview}")
                else:
                    print(f"{key}: {value}")

def format_for_gpt2_training(dataset, dataset_name):
    """
    Format dataset for GPT-2 training with proper code-explanation pairs
    """
    print(f"\nFormatting {dataset_name} for GPT-2 training...")
    
    formatted_data = []
    
    if 'train' in dataset:
        train_data = dataset['train']
        
        for i, example in enumerate(train_data):
            # Different datasets have different column names
            code = None
            explanation = None
            
            # CodeXGLUE format
            if 'code' in example and 'docstring' in example:
                code = example['code']
                explanation = example['docstring']
            
            # Alternative formats
            elif 'source_code' in example and 'target' in example:
                code = example['source_code']
                explanation = example['target']
            
            if code and explanation:
                # Clean and format the data
                formatted_example = {
                    'code': code.strip(),
                    'explanation': explanation.strip()
                }
                formatted_data.append(formatted_example)
            
            # Progress indicator
            if i % 1000 == 0:
                print(f"  Processed {i} examples...")
    
    print(f"Formatted {len(formatted_data)} examples")
    return formatted_data

def save_formatted_data(formatted_data, filename):
    """
    Save formatted data to JSON file
    """
    output_path = f"data/processed/{filename}"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(formatted_data)} examples to {output_path}")

def create_train_test_splits(data, train_ratio=0.8, val_ratio=0.1):
    """
    Create train/validation/test splits
    """
    print(f"\nCreating data splits...")
    
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    # Save splits
    splits = {
        'train': train_data,
        'validation': val_data,
        'test': test_data
    }
    
    for split_name, split_data in splits.items():
        output_path = f"data/splits/{split_name}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        print(f"{split_name}: {len(split_data)} examples saved to {output_path}")
    
    return splits

def main():
    """
    Main function to collect and prepare CodeT5 datasets
    """
    print("Starting CodeT5 dataset collection...")
    
    # Setup
    setup_directories()
    
    # CodeXGLUE data loading
    print("\n" + "="*50)
    print("CodeXGLUE Dataset")
    print("="*50)
    
    dataset = download_codexglue_dataset()
    if dataset:
        explore_dataset_structure(dataset, "CodeXGLUE")
        formatted_data = format_for_gpt2_training(dataset, "CodeXGLUE")
        if formatted_data:
            save_formatted_data(formatted_data, "codexglue_formatted.json")
            splits = create_train_test_splits(formatted_data)
            print(f"CodeXGLUE data collection completed!")
            return
    

    
    print("All methods failed. Please check your internet connection and try again.")

if __name__ == "__main__":
    main()