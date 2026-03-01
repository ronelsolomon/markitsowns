#!/usr/bin/env python3
"""
Script to export models in safetensors format
"""

import os
import subprocess
import sys
from pathlib import Path

def export_to_safetensors():
    """Export model to safetensors format"""
    
    print("Safetensors Export Guide")
    print("=" * 40)
    
    print("\n1. Current Ollama models:")
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n2. Safetensors Export Options:")
    print("   Option A: Use Hugging Face transformers")
    print("   ```python")
    print("   from transformers import AutoModelForCausalLM, AutoTokenizer")
    print("   ")
    print("   # Download the base model")
    print("   model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B')")
    print("   tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')")
    print("   ")
    print("   # Save in safetensors format")
    print("   model.save_pretrained('./llama3-safetensors', safe_serialization=True)")
    print("   tokenizer.save_pretrained('./llama3-safetensors')")
    print("   ```")
    
    print("\n   Option B: Convert from Ollama (complex)")
    print("   - Requires extracting weights from Ollama blobs")
    print("   - Need to convert to PyTorch format first")
    print("   - Then save as safetensors")
    
    print("\n3. Creating export script...")
    
    # Create export script
    export_script = '''#!/usr/bin/env python3
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
'''
    
    with open('export_to_safetensors.py', 'w') as f:
        f.write(export_script)
    
    print("✓ Created 'export_to_safetensors.py'")
    
    print("\n4. Requirements for safetensors export:")
    print("   pip install transformers torch safetensors huggingface_hub")
    print("   huggingface-cli login")
    
    print("\n5. Current files:")
    current_dir = Path('.')
    for file in current_dir.glob('*'):
        if file.is_file() and any(ext in file.name for ext in ['.gguf', '.blob', '.safetensors']):
            size = file.stat().st_size / (1024**3)  # GB
            print(f"   {file.name} ({size:.1f}GB)")
    
    print("\n6. To add your custom system prompt to safetensors:")
    print("   You'll need to fine-tune the model with your document content")
    print("   or create a RAG system that uses the document chunks")

if __name__ == "__main__":
    export_to_safetensors()
