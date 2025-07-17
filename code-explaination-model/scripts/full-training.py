"""
Full training script for GPT-2 code explanation model
"""
import os
os.environ['MallocStackLogging'] = '0'
import torch
import wandb
from pathlib import Path
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset

class CodeExplanationDataset(Dataset):
    def __init__(self, tokenized_data):
        self.input_ids = tokenized_data['input_ids']
        self.attention_mask = tokenized_data['attention_mask']
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.input_ids[idx].clone()
        }

def setup_model_and_tokenizer():
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    special_tokens = {
        'pad_token': '<pad>',
        'sep_token': '<sep>',
    }
    tokenizer.add_special_tokens(special_tokens)
    
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

def main():
    print("Starting full GPT-2 training...")
    
    # Initialize W&B
    wandb.init(
        project="code-explanation-model",
        name="gpt2-full-training-v1",
        config={
            "model": "gpt2-124m",
            "task": "code-to-explanation",
            "epochs": 3,
            "batch_size": 4,
            "learning_rate": 5e-5,
        }
    )
    
    # Setup
    model, tokenizer = setup_model_and_tokenizer()
    
    # Load full datasets
    print("Loading training data...")
    train_data = torch.load("data/tokenized/train_tokenized.pt", map_location='cpu')
    val_data = torch.load("data/tokenized/validation_tokenized.pt", map_location='cpu')
    
    train_dataset = CodeExplanationDataset(train_data)
    val_dataset = CodeExplanationDataset(val_data)
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments for full training
    training_args = TrainingArguments(
        output_dir="./models/gpt2-code-explanation",
        run_name="gpt2-full-training-v1",
        overwrite_output_dir=True,
        
        # Training parameters
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,  # Effective batch size = 4 * 2 = 8
        
        # Optimization
        learning_rate=5e-5,
        warmup_steps=500,
        weight_decay=0.01,
        
        # Logging and evaluation
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        save_total_limit=3,
        
        # Memory optimization
        dataloader_pin_memory=False,
        fp16=False,  # Set to True if you have compatible hardware
        
        # W&B
        report_to="wandb",
        logging_dir="./logs",
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
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model()
    tokenizer.save_pretrained("./models/gpt2-code-explanation")
    
    print("Training completed!")
    
    # Test final model
    test_final_model(model, tokenizer)

def test_final_model(model, tokenizer):
    """Test the trained model"""
    print("\nTesting final model...")
    
    test_examples = [
        "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
        "def is_even(num):\n    return num % 2 == 0",
        "def find_max(lst):\n    return max(lst)"
    ]
    
    model.eval()
    for code in test_examples:
        test_input = f"<code>{code}<sep>"
        inputs = tokenizer.encode(test_input, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
        if '<sep>' in generated:
            explanation = generated.split('<sep>')[-1].strip()
            print(f"\nCode: {code}")
            print(f"Explanation: {explanation}")

if __name__ == "__main__":
    main()