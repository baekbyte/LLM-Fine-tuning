import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import os

def setup_model_and_tokenizer():
    """
    Load and configure GPT-2 124M model for code-to-explanation fine-tuning
    """
    print("Setting up GPT-2 124M model...")
    
    # Load pre-trained GPT-2 124M model and tokenizer
    model_name = 'gpt2'  # This is the 124M parameter version
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Add special tokens for our task
    special_tokens = {
        'pad_token': '<pad>',
        'sep_token': '<sep>',  # To separate code from explanation
    }
    
    # Add the special tokens
    tokenizer.add_special_tokens(special_tokens)
    
    # Load model
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Resize model embeddings to accommodate new tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Print model info
    print(f"Model loaded: {model_name}")
    print(f"Model parameters: {model.num_parameters():,}")
    print(f"Vocabulary size: {len(tokenizer)}")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    return model, tokenizer, device

def format_training_example(code, explanation, tokenizer):
    """
    Format a single training example for GPT-2
    Format: <code>CODE_HERE<sep>EXPLANATION_HERE
    """
    # Create the input format
    formatted_text = f"<code>{code}<sep>{explanation}"
    
    # Tokenize
    tokens = tokenizer(
        formatted_text,
        truncation=True,
        padding=True,
        max_length=512,  # Adjust based on your data
        return_tensors='pt'
    )
    
    return tokens

def test_model_setup():
    """
    Test the model setup with a simple example
    """
    model, tokenizer, device = setup_model_and_tokenizer()
    
    # Test with a simple example
    test_code = "def add(a, b):\n    return a + b"
    test_explanation = "This function takes two parameters and returns their sum."
    
    # Format and tokenize
    tokens = format_training_example(test_code, test_explanation, tokenizer)
    
    print("\nTesting model setup:")
    print(f"Input shape: {tokens['input_ids'].shape}")
    print(f"Sample tokens: {tokens['input_ids'][0][:10]}")
    
    # Test model forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(**tokens)
        print(f"Output shape: {outputs.logits.shape}")
    
    print("Model setup test passed!")
    
    return model, tokenizer, device

if __name__ == "__main__":
    model, tokenizer, device = test_model_setup()