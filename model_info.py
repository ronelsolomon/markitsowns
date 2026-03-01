#!/usr/bin/env python3
"""
Show information about created models and document chunks
"""

import os
import subprocess
from pathlib import Path

def show_model_info():
    """Display information about Ollama models and document chunks"""
    
    print("=" * 60)
    print("OLLAMA MODELS")
    print("=" * 60)
    
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("Error listing Ollama models:", result.stderr)
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("DOCUMENT CHUNKS LOCATION")
    print("=" * 60)
    
    # Check for document chunks directory
    chunks_dir = Path("document_chunks")
    if chunks_dir.exists():
        print(f"Document chunks found in: {chunks_dir.absolute()}")
        
        # List chunk files
        chunk_files = list(chunks_dir.glob("*.txt"))
        print(f"Number of chunk files: {len(chunk_files)}")
        
        if chunk_files:
            print("\nChunk files:")
            for i, chunk_file in enumerate(sorted(chunk_files)[:10]):  # Show first 10
                size = chunk_file.stat().st_size
                print(f"  {chunk_file.name} ({size:,} bytes)")
            
            if len(chunk_files) > 10:
                print(f"  ... and {len(chunk_files) - 10} more files")
        
        # Check for index file
        index_file = chunks_dir / "index.txt"
        if index_file.exists():
            print(f"\nIndex file: {index_file.name}")
            with open(index_file, 'r') as f:
                lines = f.readlines()[:5]  # Show first 5 lines
                for line in lines:
                    print(f"  {line.strip()}")
    else:
        print("No document chunks directory found.")
    
    print("\n" + "=" * 60)
    print("OLLAMA MODEL STORAGE LOCATION")
    print("=" * 60)
    
    ollama_dir = Path.home() / ".ollama" / "models"
    if ollama_dir.exists():
        print(f"Ollama models stored in: {ollama_dir}")
        
        # List model directories
        model_dirs = [d for d in ollama_dir.iterdir() if d.is_dir()]
        print(f"Number of model directories: {len(model_dirs)}")
        
        for model_dir in model_dirs:
            print(f"  {model_dir.name}/")
    else:
        print(f"Ollama models directory not found at: {ollama_dir}")

if __name__ == "__main__":
    show_model_info()
