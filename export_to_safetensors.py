#!/usr/bin/env python3
"""
Export Llama 3 model to safetensors format
"""

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    print("Downloading Meta-Llama-3-8B from Hugging Face...")
    
    # Note: You may need to accept the license terms first
    # Visit: https://huggingface.co/meta-llama/Meta-Llama-3-8B
    
    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Meta-Llama-3-8B',
        torch_dtype=torch.float16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
    
    print("Saving to safetensors format...")
    model.save_pretrained('./llama3-safetensors', safe_serialization=True)
    tokenizer.save_pretrained('./llama3-safetensors')
    
    print("✓ Successfully saved to ./llama3-safetensors/")
    
except ImportError:
    print("❌ transformers not installed. Install with:")
    print("   pip install transformers torch safetensors")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nNote: You may need to:")
    print("1. Request access to Meta-Llama-3-8B on Hugging Face")
    print("2. Login with: huggingface-cli login")
    print("3. Accept the license terms")
