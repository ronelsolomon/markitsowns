#!/usr/bin/env python3
"""
Script to download GGUF models for local use
"""

import os
import subprocess
import sys

def download_gguf_model():
    """Download a GGUF model from various sources"""
    
    print("GGUF Model Downloader")
    print("=" * 40)
    
    # Option 1: Try Ollama's built-in download
    print("\n1. Trying to pull llama3:latest (this may create GGUF-compatible files)...")
    try:
        result = subprocess.run(['ollama', 'pull', 'llama3:latest'], 
                          capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Successfully pulled llama3:latest")
        else:
            print("✗ Failed to pull llama3:latest")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Option 2: Manual download instructions
    print("\n2. Manual GGUF Download Options:")
    print("   To get a true GGUF file, you need to:")
    print("   a) Visit: https://huggingface.co/TheBloke/Llama-3-8B-GGUF")
    print("   b) Download 'llama-3-8b.Q4_K_M.gguf' (~4.66GB)")
    print("   c) Save it as './llama-3-8b.Q4_K_M.gguf'")
    print("   d) Update your Modelfile FROM line to: FROM ./llama-3-8b.Q4_K_M.gguf")
    
    # Option 3: Create GGUF-compatible model
    print("\n3. Creating GGUF-compatible model with current setup...")
    
    # Create a new Modelfile for GGUF
    gguf_modelfile = """# GGUF-compatible Modelfile
FROM llama3:latest
SYSTEM You are a specialized assistant trained on Deep Learning textbook by Ian Goodfellow, Yoshua Bengio, and Aaron Courville. You have comprehensive knowledge of neural networks, backpropagation, optimization algorithms, convolutional networks, recurrent networks, autoencoders, representation learning, Monte Carlo methods, and practical applications. Answer questions based on this textbook content accurately and in detail.
"""
    
    with open('gguf-compatible-modelfile', 'w') as f:
        f.write(gguf_modelfile)
    
    print("✓ Created 'gguf-compatible-modelfile'")
    print("   Use: ollama create deep-learning-gguf -f gguf-compatible-modelfile")
    
    # Check current files
    print("\n4. Current files:")
    if os.path.exists('deep-learning-blob'):
        size = os.path.getsize('deep-learning-blob') / (1024**3)  # GB
        print(f"   deep-learning-blob ({size:.1f}GB) - Ollama blob format")
    
    if os.path.exists('llama-3-8b.Q4_K_M.gguf'):
        size = os.path.getsize('llama-3-8b.Q4_K_M.gguf') / (1024**3)  # GB
        print(f"   llama-3-8b.Q4_K_M.gguf ({size:.1f}GB) - True GGUF format ✓")
    
    print("\nNext steps:")
    print("1. Download GGUF file manually from Hugging Face (if needed)")
    print("2. Update Modelfile FROM line to point to GGUF file")
    print("3. Create new model: ollama create model-name -f modelfile")

if __name__ == "__main__":
    download_gguf_model()
