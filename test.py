# MacBook Air Optimized LLM Code
# ==============================

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import time
import psutil
import os

# Check system resources
def check_system():
    """Check system specifications"""
    print("System Information")
    print("=" * 30)
    
    # CPU info
    print(f"CPU cores: {os.cpu_count()}")
    print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # Check for MPS (Apple Silicon GPU)
    if torch.backends.mps.is_available():
        print("MPS (Apple Silicon GPU) available")
        device = torch.device("mps")
    else:
        print("Using CPU only")
        device = torch.device("cpu")
    
    return device

# Recommended models for MacBook Air
def load_lightweight_model(model_name="distilgpt2"):
    """Load a model optimized for MacBook Air"""
    
    print(f"Loading lightweight model: {model_name}")
    
    # Recommended models for MacBook Air
    recommended_models = {
        "distilgpt2": "Smaller, faster GPT-2",
        "gpt2": "Original GPT-2 (124M params)",
        "microsoft/DialoGPT-small": "Small conversational model",
        "google/flan-t5-small": "Small instruction-following model"
    }
    
    if model_name in recommended_models:
        print(f"Good choice: {recommended_models[model_name]}")
    else:
        print("This model might be too large for MacBook Air")
    
    # Load with CPU optimization
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for CPU
        low_cpu_mem_usage=True      # Reduce memory usage
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = check_system()
    model = model.to(device)
    
    # Get model size
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"Estimated model size: {param_count * 4 / (1024**3):.2f} GB")
    
    return model, tokenizer, device

# Memory-efficient generation
def efficient_generation(model, tokenizer, device, prompt, max_length=50):
    """Generate text efficiently on MacBook Air"""
    
    print(f"Generating text for: '{prompt}'")
    start_time = time.time()
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate with memory optimization
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            use_cache=True  # Use KV cache for efficiency
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generation_time = time.time() - start_time
    
    print(f"Generated: {generated_text}")
    print(f"Generation time: {generation_time:.2f} seconds")
    print(f"Memory usage: {psutil.virtual_memory().percent}%")
    
    return generated_text

# Batch processing with smaller batches
def small_batch_processing(model, tokenizer, device):
    """Process multiple prompts with small batches"""
    
    prompts = [
        "The future of AI is",
        "Machine learning helps",
        "Technology will enable"
    ]
    
    print("Small Batch Processing")
    print("=" * 30)
    
    results = []
    
    # Process one by one to avoid memory issues
    for i, prompt in enumerate(prompts, 1):
        print(f"Processing prompt {i}/{len(prompts)}")
        
        result = efficient_generation(model, tokenizer, device, prompt, max_length=40)
        results.append(result)
        
        # Small delay to prevent overheating
        time.sleep(0.5)
    
    return results

# Pipeline approach (often more efficient on CPU)
def pipeline_approach():
    """Use pipeline for better CPU optimization"""
    
    print("Pipeline Approach (CPU Optimized)")
    print("=" * 40)
    
    # Pipeline automatically optimizes for CPU
    generator = pipeline(
        'text-generation',
        model='distilgpt2',
        device=-1,  # Force CPU
        torch_dtype=torch.float32
    )
    
    prompts = [
        "The best thing about AI is",
        "In the future, computers will",
        "The most important skill for programmers is"
    ]
    
    for prompt in prompts:
        start_time = time.time()
        
        result = generator(
            prompt,
            max_length=50,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
        
        generation_time = time.time() - start_time
        
        print(f"Prompt: {prompt}")
        print(f"Result: {result[0]['generated_text']}")
        print(f"Time: {generation_time:.2f}s")
        print()

# Memory monitoring
def monitor_memory():
    """Monitor system memory usage"""
    
    memory = psutil.virtual_memory()
    
    print("Memory Status")
    print("=" * 20)
    print(f"Used: {memory.percent}%")
    print(f"Available: {memory.available / (1024**3):.1f} GB")
    print(f"Free: {memory.free / (1024**3):.1f} GB")
    
    if memory.percent > 85:
        print("High memory usage - consider using smaller models")
    elif memory.percent > 70:
        print("Moderate memory usage - monitor closely")
    else:
        print("Memory usage is healthy")

# Quick test function
def quick_test():
    """Quick test to see if everything works"""
    
    print("Quick Test")
    print("=" * 15)
    
    # Use the most efficient approach
    try:
        generator = pipeline('text-generation', model='distilgpt2', device=-1)
        result = generator("Hello, AI is", max_length=30, num_return_sequences=1)
        print(f"Test successful: {result[0]['generated_text']}")
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False

# Main execution optimized for MacBook Air
if __name__ == "__main__":
    print("MacBook Air LLM Tutorial")
    print("=" * 40)
    
    # Quick test first
    if not quick_test():
        print("Please install required packages: pip install transformers torch")
        exit()
    
    print("\n" + "=" * 40)
    
    # Check system
    device = check_system()
    
    print("\n" + "=" * 40)
    
    # Monitor memory
    monitor_memory()
    
    print("\n" + "=" * 40)
    
    # Use pipeline approach (most efficient for CPU)
    pipeline_approach()
    
    print("\n" + "=" * 40)
    
    # Optional: Direct model usage (comment out if too slow)
    # model, tokenizer, device = load_lightweight_model("distilgpt2")
    # small_batch_processing(model, tokenizer, device)
    
    print("\nMacBook Air tutorial completed!")
    print("Tip: Use pipeline approach for best CPU performance")