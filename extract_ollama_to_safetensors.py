#!/usr/bin/env python3
"""
Extract Ollama model and convert to safetensors format
"""

import json
import shutil
import os
from pathlib import Path
import subprocess

def extract_ollama_model():
    """Extract model from Ollama storage and convert to safetensors"""
    
    print("Extracting Ollama Model to Safetensors")
    print("=" * 50)
    
    # Paths
    ollama_models = Path.home() / ".ollama" / "models"
    manifests_dir = ollama_models / "manifests" / "registry.ollama.ai" / "library"
    blobs_dir = ollama_models / "blobs"
    
    # Model info
    model_name = "deep-learning-complete"
    model_manifest_path = manifests_dir / model_name / "latest"
    
    if not model_manifest_path.exists():
        print(f"❌ Model manifest not found: {model_manifest_path}")
        return
    
    # Read manifest
    with open(model_manifest_path, 'r') as f:
        manifest = json.load(f)
    
    print(f"✓ Found model: {model_name}")
    print(f"✓ Model layers: {len(manifest['layers'])}")
    
    # Find the main model layer
    model_layer = None
    for layer in manifest['layers']:
        if layer['mediaType'] == 'application/vnd.ollama.image.model':
            model_layer = layer
            break
    
    if not model_layer:
        print("❌ Model layer not found")
        return
    
    # Get the blob file
    blob_digest = model_layer['digest'].replace('sha256:', '')
    # Try different blob path structures
    possible_paths = [
        blobs_dir / "sha256" / blob_digest,
        blobs_dir / f"sha256-{blob_digest}",
        blobs_dir / blob_digest
    ]
    
    blob_path = None
    for path in possible_paths:
        if path.exists():
            blob_path = path
            break
    
    if not blob_path:
        print(f"❌ Model blob not found with digest: {blob_digest}")
        print("   Searched paths:")
        for path in possible_paths:
            print(f"     {path}")
        return
    
    print(f"✓ Found model blob: {blob_path}")
    print(f"✓ Blob size: {model_layer['size'] / (1024**3):.1f} GB")
    
    # Copy blob to current directory
    output_file = Path(f"{model_name}-extracted.gguf")
    shutil.copy2(blob_path, output_file)
    
    print(f"✓ Copied to: {output_file}")
    
    # Try to identify format
    try:
        result = subprocess.run(['file', str(output_file)], 
                          capture_output=True, text=True)
        print(f"✓ File type: {result.stdout.strip()}")
    except:
        print("⚠️  Could not determine file type")
    
    # Check if it's already GGUF
    with open(output_file, 'rb') as f:
        header = f.read(4)
        if header == b'GGUF':
            print("✓ File is already in GGUF format!")
        else:
            print("⚠️  File is in Ollama's internal format")
            print("   Converting to standard formats would require:")
            print("   1. Extracting weights using Ollama's internal tools")
            print("   2. Converting to PyTorch format")
            print("   3. Saving as safetensors")
            print("   This is complex and may not be reliable")
    
    # Create conversion instructions
    conversion_script = f'''#!/bin/bash
# Conversion script for {model_name}

echo "Converting {model_name} to different formats..."

# Option 1: Use with llama.cpp directly
echo "1. Using with llama.cpp:"
echo "   ./main -m {output_file} -p \"Your prompt here\""

# Option 2: Try to convert with transformers (experimental)
echo "2. Converting to safetensors (experimental):"
echo "   This would require custom conversion tools"

# Option 3: Keep using Ollama (recommended)
echo "3. Continue using Ollama:"
echo "   ollama run {model_name}"
echo "   python chat_with_model.py {model_name}"
'''
    
    with open(f'convert_{model_name}.sh', 'w') as f:
        f.write(conversion_script)
    
    os.chmod(f'convert_{model_name}.sh', 0o755)
    print(f"✓ Created conversion script: convert_{model_name}.sh")
    
    print(f"\n📁 Summary:")
    print(f"   Original: {blob_path}")
    print(f"   Extracted: {output_file}")
    print(f"   Size: {model_layer['size'] / (1024**3):.1f} GB")
    print(f"   Format: Ollama internal (can be used with llama.cpp)")

if __name__ == "__main__":
    extract_ollama_model()
