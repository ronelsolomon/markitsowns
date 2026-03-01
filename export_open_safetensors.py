#!/usr/bin/env python3
"""
Export open models to safetensors format (no access required)
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def export_open_model():
    """Export an open model to safetensors"""
    
    print("Exporting Open Model to Safetensors")
    print("=" * 40)
    
    # Use an open model that doesn't require access
    model_name = "microsoft/DialoGPT-medium"  # Or "EleutherAI/gpt-neo-125M"
    
    print(f"Downloading {model_name}...")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        output_dir = f"./{model_name.split('/')[-1]}-safetensors"
        print(f"Saving to {output_dir}/")
        
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        
        print(f"✓ Successfully saved to {output_dir}/")
        print(f"✓ Files saved in safetensors format")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    export_open_model()
