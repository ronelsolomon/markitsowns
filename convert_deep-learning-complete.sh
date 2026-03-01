#!/bin/bash
# Conversion script for deep-learning-complete

echo "Converting deep-learning-complete to different formats..."

# Option 1: Use with llama.cpp directly
echo "1. Using with llama.cpp:"
echo "   ./main -m deep-learning-complete-extracted.gguf -p "Your prompt here""

# Option 2: Try to convert with transformers (experimental)
echo "2. Converting to safetensors (experimental):"
echo "   This would require custom conversion tools"

# Option 3: Keep using Ollama (recommended)
echo "3. Continue using Ollama:"
echo "   ollama run deep-learning-complete"
echo "   python chat_with_model.py deep-learning-complete"
