"""
Data Quality Verification and Preprocessing Pipeline for CodeT5 Dataset
Handles verification, cleaning, and tokenization for GPT-2 training
"""

import json
import re
import ast
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

# For tokenization
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

class DataQualityVerifier:
    """Verify and analyze data quality for code-explanation pairs"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = self.load_data()
        self.quality_report = {}
        
    def load_data(self) -> List[Dict]:
        """Load data from JSON file"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} examples from {self.data_path}")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return []
    
    def check_basic_structure(self) -> Dict:
        """Check basic structure of data"""
        print("\nChecking basic data structure...")
        
        total_examples = len(self.data)
        valid_examples = 0
        missing_code = 0
        missing_explanation = 0
        empty_code = 0
        empty_explanation = 0
        
        for i, example in enumerate(self.data):
            if not isinstance(example, dict):
                continue
                
            has_code = 'code' in example and example['code'] is not None
            has_explanation = 'explanation' in example and example['explanation'] is not None
            
            if not has_code:
                missing_code += 1
            elif not example['code'].strip():
                empty_code += 1
                
            if not has_explanation:
                missing_explanation += 1
            elif not example['explanation'].strip():
                empty_explanation += 1
                
            if has_code and has_explanation and example['code'].strip() and example['explanation'].strip():
                valid_examples += 1
        
        structure_report = {
            'total_examples': total_examples,
            'valid_examples': valid_examples,
            'missing_code': missing_code,
            'missing_explanation': missing_explanation,
            'empty_code': empty_code,
            'empty_explanation': empty_explanation,
            'valid_percentage': (valid_examples / total_examples * 100) if total_examples > 0 else 0
        }
        
        print(f"Total examples: {total_examples}")
        print(f"Valid examples: {valid_examples} ({structure_report['valid_percentage']:.1f}%)")
        print(f"Missing code: {missing_code}")
        print(f"Missing explanation: {missing_explanation}")
        print(f"Empty code: {empty_code}")
        print(f"Empty explanation: {empty_explanation}")
        
        return structure_report
    
    def check_code_quality(self) -> Dict:
        """Check quality of code examples"""
        print("\nChecking code quality...")
        
        syntactically_valid = 0
        has_functions = 0
        has_classes = 0
        has_imports = 0
        too_short = 0
        too_long = 0
        
        code_lengths = []
        
        for example in self.data:
            if not isinstance(example, dict) or 'code' not in example:
                continue
                
            code = example['code']
            if not code or not code.strip():
                continue
                
            code_length = len(code.split('\n'))
            code_lengths.append(code_length)
            
            # Check if code is too short or too long
            if code_length < 2:
                too_short += 1
            elif code_length > 50:
                too_long += 1
            
            # Check syntax validity
            try:
                ast.parse(code)
                syntactically_valid += 1
            except SyntaxError:
                pass
            
            # Check for functions and classes
            if 'def ' in code:
                has_functions += 1
            if 'class ' in code:
                has_classes += 1
            if 'import ' in code or 'from ' in code:
                has_imports += 1
        
        code_report = {
            'syntactically_valid': syntactically_valid,
            'has_functions': has_functions,
            'has_classes': has_classes,
            'has_imports': has_imports,
            'too_short': too_short,
            'too_long': too_long,
            'avg_code_length': np.mean(code_lengths) if code_lengths else 0,
            'median_code_length': np.median(code_lengths) if code_lengths else 0
        }
        
        print(f"Syntactically valid: {syntactically_valid}")
        print(f"Contains functions: {has_functions}")
        print(f"Contains classes: {has_classes}")
        print(f"Contains imports: {has_imports}")
        print(f"Too short (<2 lines): {too_short}")
        print(f"Too long (>50 lines): {too_long}")
        print(f"Average code length: {code_report['avg_code_length']:.1f} lines")
        
        return code_report
    
    def check_explanation_quality(self) -> Dict:
        """Check quality of explanations"""
        print("\nChecking explanation quality...")
        
        explanation_lengths = []
        has_technical_terms = 0
        too_generic = 0
        too_short_explanations = 0
        too_long_explanations = 0
        
        # Common technical terms to look for
        technical_terms = ['function', 'method', 'class', 'variable', 'parameter', 'return', 'loop', 'condition', 'algorithm']
        generic_phrases = ['this code', 'this function', 'this method', 'the code', 'the function']
        
        for example in self.data:
            if not isinstance(example, dict) or 'explanation' not in example:
                continue
                
            explanation = example['explanation']
            if not explanation or not explanation.strip():
                continue
                
            explanation_lower = explanation.lower()
            word_count = len(explanation.split())
            explanation_lengths.append(word_count)
            
            # Check explanation length
            if word_count < 5:
                too_short_explanations += 1
            elif word_count > 100:
                too_long_explanations += 1
            
            # Check for technical terms
            if any(term in explanation_lower for term in technical_terms):
                has_technical_terms += 1
            
            # Check for generic phrases
            if any(phrase in explanation_lower for phrase in generic_phrases):
                too_generic += 1
        
        explanation_report = {
            'has_technical_terms': has_technical_terms,
            'too_generic': too_generic,
            'too_short_explanations': too_short_explanations,
            'too_long_explanations': too_long_explanations,
            'avg_explanation_length': np.mean(explanation_lengths) if explanation_lengths else 0,
            'median_explanation_length': np.median(explanation_lengths) if explanation_lengths else 0
        }
        
        print(f"Has technical terms: {has_technical_terms}")
        print(f"Too generic: {too_generic}")
        print(f"Too short (<5 words): {too_short_explanations}")
        print(f"Too long (>100 words): {too_long_explanations}")
        print(f"Average explanation length: {explanation_report['avg_explanation_length']:.1f} words")
        
        return explanation_report
    
    def generate_quality_report(self) -> Dict:
        """Generate comprehensive quality report"""
        print("Generating comprehensive quality report...")
        
        structure_report = self.check_basic_structure()
        code_report = self.check_code_quality()
        explanation_report = self.check_explanation_quality()
        
        # Show sample good and bad examples
        self.show_sample_examples()
        
        self.quality_report = {
            'structure': structure_report,
            'code': code_report,
            'explanation': explanation_report
        }
        
        return self.quality_report
    
    def show_sample_examples(self, num_samples=3):
        """Show sample examples for manual inspection"""
        print(f"\nSample examples for manual inspection:")
        
        valid_examples = [ex for ex in self.data if isinstance(ex, dict) and 
                         'code' in ex and 'explanation' in ex and 
                         ex['code'].strip() and ex['explanation'].strip()]
        
        if len(valid_examples) < num_samples:
            num_samples = len(valid_examples)
        
        for i in range(num_samples):
            example = valid_examples[i]
            print(f"\n--- Example {i+1} ---")
            print(f"Code:\n{example['code']}")
            print(f"Explanation: {example['explanation']}")
            print("-" * 40)
    
    def save_quality_report(self, output_path: str):
        """Save quality report to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.quality_report, f, indent=2, ensure_ascii=False)
        print(f"Quality report saved to {output_path}")


class DataCleaner:
    """Clean and preprocess data for training"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = self.load_data()
        self.original_count = len(self.data)
        
    def load_data(self) -> List[Dict]:
        """Load data from JSON file"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} examples for cleaning")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return []
    
    def remove_invalid_examples(self) -> List[Dict]:
        """Remove examples with missing or empty code/explanation"""
        print("\n🧹 Removing invalid examples...")
        
        valid_data = []
        for example in self.data:
            if (isinstance(example, dict) and 
                'code' in example and 'explanation' in example and
                example['code'] and example['explanation'] and
                example['code'].strip() and example['explanation'].strip()):
                valid_data.append(example)
        
        removed = len(self.data) - len(valid_data)
        print(f"  Removed {removed} invalid examples")
        return valid_data
    
    def clean_code(self, code: str) -> str:
        """Clean code formatting"""
        if not code:
            return ""
        
        # Remove excessive whitespace
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove trailing whitespace
            cleaned_line = line.rstrip()
            cleaned_lines.append(cleaned_line)
        
        # Remove empty lines at the beginning and end
        while cleaned_lines and not cleaned_lines[0].strip():
            cleaned_lines.pop(0)
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()
        
        return '\n'.join(cleaned_lines)
    
    def clean_explanation(self, explanation: str) -> str:
        """Clean explanation text"""
        if not explanation:
            return ""
        
        # Remove excessive whitespace
        cleaned = ' '.join(explanation.split())
        
        # Remove common prefixes/suffixes
        prefixes_to_remove = ['Summary:', 'Description:', 'Docstring:', 'Doc:']
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        # Ensure explanation ends with proper punctuation
        if cleaned and not cleaned.endswith('.'):
            cleaned += '.'
        
        return cleaned
    
    def filter_by_quality(self, data: List[Dict]) -> List[Dict]:
        """Filter examples based on quality criteria"""
        print("\nFiltering by quality criteria...")
        
        filtered_data = []
        
        for example in data:
            code = example['code']
            explanation = example['explanation']
            
            # Skip if code is too short or too long
            code_lines = len(code.split('\n'))
            if code_lines < 2 or code_lines > 50:
                continue
            
            # Skip if explanation is too short or too long
            explanation_words = len(explanation.split())
            if explanation_words < 5 or explanation_words > 100:
                continue
            
            # Skip if code is not syntactically valid
            try:
                ast.parse(code)
            except SyntaxError:
                continue
            
            # Skip very generic explanations
            explanation_lower = explanation.lower()
            if explanation_lower.count('this') > 3:  # Too many generic references
                continue
            
            filtered_data.append(example)
        
        removed = len(data) - len(filtered_data)
        print(f"Removed {removed} low-quality examples")
        return filtered_data
    
    def clean_all_data(self) -> List[Dict]:
        """Apply all cleaning operations"""
        print("Starting data cleaning process...")
        
        # Remove invalid examples
        valid_data = self.remove_invalid_examples()
        
        # Clean individual fields
        cleaned_data = []
        for example in valid_data:
            cleaned_example = {
                'code': self.clean_code(example['code']),
                'explanation': self.clean_explanation(example['explanation'])
            }
            cleaned_data.append(cleaned_example)
        
        # Filter by quality
        final_data = self.filter_by_quality(cleaned_data)
        
        print(f"Cleaning complete: {self.original_count} → {len(final_data)} examples")
        return final_data
    
    def save_cleaned_data(self, cleaned_data: List[Dict], output_path: str):
        """Save cleaned data to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
        print(f"Cleaned data saved to {output_path}")


class DataTokenizer:
    """Tokenize data for GPT-2 training"""
    
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # Add padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def format_for_gpt2(self, code: str, explanation: str) -> str:
        """Format code-explanation pair for GPT-2 training"""
        # Use special tokens to separate code and explanation
        formatted_text = f"<|code|>{code}<|explanation|>{explanation}<|endoftext|>"
        return formatted_text
    
    def add_special_tokens(self):
        """Add special tokens to tokenizer"""
        special_tokens = ["<|code|>", "<|explanation|>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        print(f"Added special tokens: {special_tokens}")
    
    def tokenize_dataset(self, data: List[Dict], max_length: int = 512) -> Dict:
        """Tokenize entire dataset"""
        print(f"\nTokenizing dataset with max_length={max_length}...")
        
        # Add special tokens
        self.add_special_tokens()
        
        formatted_texts = []
        for example in data:
            formatted_text = self.format_for_gpt2(example['code'], example['explanation'])
            formatted_texts.append(formatted_text)
        
        # Tokenize all texts
        tokenized = self.tokenizer(
            formatted_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Create dataset compatible with Hugging Face
        tokenized_dataset = {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': tokenized['input_ids'].clone()  # For language modeling
        }
        
        print(f"Tokenized {len(data)} examples")
        print(f"Input shape: {tokenized_dataset['input_ids'].shape}")
        print(f"Vocabulary size: {len(self.tokenizer)}")
        
        return tokenized_dataset
    
    def analyze_token_lengths(self, data: List[Dict]) -> Dict:
        """Analyze token length distribution"""
        print("\nAnalyzing token length distribution...")
        
        token_lengths = []
        for example in data:
            formatted_text = self.format_for_gpt2(example['code'], example['explanation'])
            tokens = self.tokenizer.encode(formatted_text)
            token_lengths.append(len(tokens))
        
        stats = {
            'min_length': min(token_lengths),
            'max_length': max(token_lengths),
            'mean_length': np.mean(token_lengths),
            'median_length': np.median(token_lengths),
            'std_length': np.std(token_lengths)
        }
        
        print(f"Min length: {stats['min_length']}")
        print(f"Max length: {stats['max_length']}")
        print(f"Mean length: {stats['mean_length']:.1f}")
        print(f"Median length: {stats['median_length']:.1f}")
        print(f"Std deviation: {stats['std_length']:.1f}")
        
        # Show percentiles
        percentiles = [50, 75, 90, 95, 99]
        for p in percentiles:
            value = np.percentile(token_lengths, p)
            print(f"{p}th percentile: {value:.0f}")
        
        return stats
    
    def save_tokenized_data(self, tokenized_dataset: Dict, output_path: str):
        """Save tokenized dataset"""
        torch.save(tokenized_dataset, output_path)
        print(f"Tokenized data saved to {output_path}")


def main():
    """Main pipeline for data verification, cleaning, and tokenization"""
    print("Starting Data Quality Verification and Preprocessing Pipeline")
    
    # Configuration
    input_files = [
        "data/splits/train.json",
        "data/splits/validation.json", 
        "data/splits/test.json"
    ]
    
    # Create output directories
    Path("data/cleaned").mkdir(exist_ok=True)
    Path("data/tokenized").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    
    # Process each split
    for input_file in input_files:
        if not Path(input_file).exists():
            print(f"File not found: {input_file}, skipping...")
            continue
        
        split_name = Path(input_file).stem
        print(f"\n{'='*60}")
        print(f"Processing {split_name.upper()} split")
        print(f"{'='*60}")
        
        # Step 1: Verify data quality
        print("\nSTEP 1: Data Quality Verification")
        verifier = DataQualityVerifier(input_file)
        quality_report = verifier.generate_quality_report()
        verifier.save_quality_report(f"reports/{split_name}_quality_report.json")
        
        # Step 2: Clean data
        print("\nSTEP 2: Data Cleaning")
        cleaner = DataCleaner(input_file)
        cleaned_data = cleaner.clean_all_data()
        cleaned_file = f"data/cleaned/{split_name}_cleaned.json"
        cleaner.save_cleaned_data(cleaned_data, cleaned_file)
        
        # Step 3: Tokenize data
        print("\nSTEP 3: Tokenization")
        tokenizer = DataTokenizer()
        token_stats = tokenizer.analyze_token_lengths(cleaned_data)
        tokenized_dataset = tokenizer.tokenize_dataset(cleaned_data)
        tokenizer.save_tokenized_data(tokenized_dataset, f"data/tokenized/{split_name}_tokenized.pt")
        
        print(f"{split_name} processing complete!")
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()