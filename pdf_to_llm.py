#!/usr/bin/env python3
"""
PDF to LLM Processor using Microsoft MarkItDown

This script converts PDF files to text using Microsoft's MarkItDown library
and then processes the extracted text with an LLM model.
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from markitdown import MarkItDown

# Load environment variables
load_dotenv()

class PDFToLLMProcessor:
    def __init__(self, model="gpt-4-turbo-preview"):
        """Initialize the processor with OpenAI client and MarkItDown."""
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.markitdown = MarkItDown()
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF using MarkItDown."""
        try:
            result = self.markitdown.convert(str(pdf_path))
            return result.text_content
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def process_with_llm(self, text, prompt="Summarize this document:"):
        """Process extracted text with LLM."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that processes document text."},
                    {"role": "user", "content": f"{prompt}\n\n{text}"}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error processing with LLM: {str(e)}")
    
    def process_pdf(self, pdf_path, prompt=None):
        """Complete pipeline: PDF -> Text -> LLM -> Result."""
        print(f"Processing PDF: {pdf_path}")
        
        # Extract text
        print("Extracting text from PDF...")
        extracted_text = self.extract_text_from_pdf(pdf_path)
        print(f"Extracted {len(extracted_text)} characters")
        
        # Process with LLM
        if prompt is None:
            prompt = "Please analyze this document and provide a comprehensive summary, including key points, main arguments, and any important insights."
        
        print("Processing with LLM...")
        result = self.process_with_llm(extracted_text, prompt)
        
        return result
    
    def save_result(self, result, output_path):
        """Save the LLM result to a file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"Result saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert PDF to text and process with LLM')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--prompt', '-p', help='Custom prompt for the LLM')
    parser.add_argument('--output', '-o', help='Output file for the result')
    parser.add_argument('--model', '-m', default='gpt-4-turbo-preview', help='LLM model to use')
    
    args = parser.parse_args()
    
    # Validate PDF path
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    if not pdf_path.suffix.lower() == '.pdf':
        print(f"Error: File is not a PDF: {pdf_path}")
        sys.exit(1)
    
    try:
        # Initialize processor
        processor = PDFToLLMProcessor(model=args.model)
        
        # Process PDF
        result = processor.process_pdf(pdf_path, args.prompt)
        
        # Output result
        if args.output:
            output_path = Path(args.output)
            processor.save_result(result, output_path)
        else:
            print("\n" + "="*50)
            print("LLM PROCESSING RESULT:")
            print("="*50)
            print(result)
            print("="*50)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
