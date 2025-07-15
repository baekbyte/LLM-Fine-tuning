"""
Quick training test with small data subset
"""

import os

import torch
import json
from pathlib import Path
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset
import numpy as np

class CodeExplanationDataset(Dataset):
    """Custom dataset for code-explanation pairs"""
    
    def __init__(self, tokenized_data, device):
        self.input_ids = tokenized_data['input_ids'].to(device)
        self.attention_mask = tokenized_data['attention_mask'].to(device)
        self.device = device
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.input_ids[idx].clone()
        }

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device('cpu')  # Use CPU instead of MPS for stability
        print("MPS available but using CPU for stability")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def load_small_subset(tokenized_file_path, subset_size=100):
    """Load a small subset of tokenized data for testing"""
    print(f"Loading small subset from {tokenized_file_path}...")
    
    # Load full tokenized data
    full_data = torch.load(tokenized_file_path, map_location='cpu')  # Load to CPU first
    
    # Take only first subset_size examples
    subset_data = {
        'input_ids': full_data['input_ids'][:subset_size],
        'attention_mask': full_data['attention_mask'][:subset_size]
    }
    
    print(f"Loaded subset: {subset_data['input_ids'].shape}")
    return subset_data

def setup_model_and_tokenizer(device):
    """Setup model and tokenizer with proper device handling"""
    print("Setting up model and tokenizer...")
    
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Add special tokens
    special_tokens = {
        'pad_token': '<pad>',
        'sep_token': '<sep>',
    }
    tokenizer.add_special_tokens(special_tokens)
    
    # Load model
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    
    # Move model to device
    model = model.to(device)
    print(f"Model moved to device: {device}")
    
    return model, tokenizer

def test_training():
    """Run a quick training test"""
    print("Starting training test...")
    
    # Get device
    device = get_device()
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(device)
    
    # Load small training subset
    train_data = load_small_subset("data/tokenized/train_tokenized.pt", subset_size=100)
    val_data = load_small_subset("data/tokenized/validation_tokenized.pt", subset_size=20)
    
    # Create datasets (don't move to device here, let trainer handle it)
    train_dataset = CodeExplanationDataset(train_data, torch.device('cpu'))
    val_dataset = CodeExplanationDataset(val_data, torch.device('cpu'))
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments - use CPU for stability
    training_args = TrainingArguments(
        output_dir="./test_results",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=5e-5,
        warmup_steps=10,
        logging_steps=5,
        save_steps=50,
        eval_strategy="steps",
        eval_steps=10,
        save_total_limit=1,
        prediction_loss_only=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        use_cpu=True,  # Force CPU usage for stability
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Test training
    print("Starting test training...")
    try:
        trainer.train()
        print("Test training completed successfully!")
        
        # Test evaluation
        eval_results = trainer.evaluate()
        print(f"Evaluation results: {eval_results}")
        
        # Test generation
        test_generation(model, tokenizer, device)
        
    except Exception as e:
        print(f"Training test failed: {e}")
        raise e

def test_generation(model, tokenizer, device):
    """Test model generation with proper device handling"""
    print("\nTesting model generation...")
    
    model.eval()
    
    # Test input
    test_code = "def add_numbers(a, b):\n    return a + b"
    test_input = f"<code>{test_code}<sep>"
    
    # Tokenize and move to same device as model
    inputs = tokenizer.encode(test_input, return_tensors='pt').to(device)
    
    # Generate
    with torch.no_grad():
        try:
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
            print(f"Generated: {generated_text}")
            
            # Extract explanation
            if '<sep>' in generated_text:
                explanation = generated_text.split('<sep>')[-1].strip()
                print(f"Explanation: {explanation}")
                
        except Exception as e:
            print(f"Generation failed: {e}")
            print("This is normal for an untrained model - the important thing is that training worked!")

if __name__ == "__main__":
    # Check if tokenized data exists
    train_file = Path("data/tokenized/train_tokenized.pt")
    val_file = Path("data/tokenized/validation_tokenized.pt")
    
    if not train_file.exists() or not val_file.exists():
        print("Tokenized data not found!")
        print("Make sure you have:")
        print("  - data/tokenized/train_tokenized.pt")
        print("  - data/tokenized/validation_tokenized.pt")
        exit(1)
    
    test_training()