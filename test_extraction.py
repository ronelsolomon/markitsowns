#!/usr/bin/env python3
"""
Test MarkItDown text extraction capabilities and limits
"""

from markitdown import MarkItDown
import sys

def test_pdf_extraction():
    """Test PDF text extraction and show statistics."""
    
    # Initialize MarkItDown
    md = MarkItDown()
    
    print("Testing MarkItDown PDF Extraction...")
    print("=" * 50)
    
    try:
        # Extract text from the PDF
        result = md.convert('Deep Learning by Ian Goodfellow, Yoshua Bengio, Aaron Courville.pdf')
        text = result.text_content
        
        # Calculate statistics
        char_count = len(text)
        word_count = len(text.split())
        line_count = len(text.splitlines())
        
        print(f"PDF File Size: 18MB")
        print(f"Extracted Characters: {char_count:,}")
        print(f"Extracted Words: {word_count:,}")
        print(f"Extracted Lines: {line_count:,}")
        print()
        
        # Show text density
        if char_count > 0:
            chars_per_mb = char_count / 18
            print(f"Text Density: {chars_per_mb:,.0f} characters per MB")
            print()
        
        # Show sample text
        print("First 300 characters:")
        print("-" * 40)
        print(text[:300])
        print("-" * 40)
        print()
        
        print("Last 300 characters:")
        print("-" * 40)
        print(text[-300:])
        print("-" * 40)
        print()
        
        # Check for common issues
        issues = []
        if char_count < 1000:
            issues.append("Very low text extraction - possible OCR issue")
        if word_count < 100:
            issues.append("Very low word count - extraction may have failed")
        
        if issues:
            print("Potential Issues:")
            for issue in issues:
                print(f"  ⚠️  {issue}")
        else:
            print("✅ Text extraction appears successful!")
        
        return text
        
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        return None

def check_llm_limits(text):
    """Check if extracted text fits within LLM context limits."""
    
    if not text:
        return
    
    print("\n" + "=" * 50)
    print("LLM Context Window Analysis")
    print("=" * 50)
    
    # Common LLM context limits
    models = {
        "GPT-3.5 Turbo": 16385,
        "GPT-4": 8192,
        "GPT-4 Turbo": 128000,
        "GPT-4o": 128000,
        "Claude 3 Sonnet": 200000,
        "Llama 3.1 8B": 128000,
        "Llama 3.1 70B": 128000,
        "Mistral 7B": 32768,
        "Qwen 2.5 7B": 32768
    }
    
    char_count = len(text)
    token_estimate = char_count // 4  # Rough estimate: 1 token ≈ 4 characters
    
    print(f"Extracted text: {char_count:,} characters")
    print(f"Estimated tokens: {token_estimate:,}")
    print()
    
    print("Model Compatibility:")
    for model, context_limit in models.items():
        if token_estimate < context_limit:
            status = "✅ Fits"
            remaining = context_limit - token_estimate
            print(f"  {model:<20} {status:<8} ({remaining:,} tokens remaining)")
        else:
            status = "❌ Too large"
            overflow = token_estimate - context_limit
            print(f"  {model:<20} {status:<8} (exceeds by {overflow:,} tokens)")
    
    print()
    
    # Recommendations
    if token_estimate > 128000:
        print("📋 Recommendations:")
        print("  • Consider chunking the PDF into sections")
        print("  • Use a model with larger context window")
        print("  • Extract specific chapters/pages only")
        print("  • Create targeted prompts for specific analysis")

if __name__ == "__main__":
    text = test_pdf_extraction()
    check_llm_limits(text)
