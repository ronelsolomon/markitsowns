#!/usr/bin/env python3
"""
Batch PDF to LLM Processor with Ollama Support

Process multiple PDF files in a directory using MarkItDown and LLM (OpenAI or Ollama).
"""

import os
import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from pdf_to_llm_ollama import PDFToLLMProcessor

class BatchPDFProcessor:
    def __init__(self, provider=None, model=None, max_workers=3):
        """Initialize batch processor."""
        self.processor = PDFToLLMProcessor(provider=provider, model=model)
        self.max_workers = max_workers
    
    def process_single_pdf(self, pdf_path, output_dir, prompt=None):
        """Process a single PDF file."""
        try:
            result = self.processor.process_pdf(pdf_path, prompt)
            
            # Generate output filename
            output_filename = pdf_path.stem + f"_{self.processor.provider}_processed.txt"
            output_path = output_dir / output_filename
            
            # Save result
            self.processor.save_result(result, output_path)
            
            return {
                'status': 'success',
                'pdf_path': str(pdf_path),
                'output_path': str(output_path),
                'provider': self.processor.provider,
                'model': self.processor.model,
                'error': None
            }
        except Exception as e:
            return {
                'status': 'error',
                'pdf_path': str(pdf_path),
                'output_path': None,
                'provider': self.processor.provider,
                'model': self.processor.model,
                'error': str(e)
            }
    
    def process_directory(self, input_dir, output_dir, prompt=None, parallel=True):
        """Process all PDFs in a directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all PDF files
        pdf_files = list(input_path.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {input_dir}")
            return []
        
        print(f"Found {len(pdf_files)} PDF files to process")
        print(f"Using LLM provider: {self.processor.provider}")
        print(f"Using model: {self.processor.model}")
        
        results = []
        
        if parallel and len(pdf_files) > 1:
            # Process in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_pdf = {
                    executor.submit(self.process_single_pdf, pdf, output_path, prompt): pdf
                    for pdf in pdf_files
                }
                
                for future in as_completed(future_to_pdf):
                    pdf = future_to_pdf[future]
                    try:
                        result = future.result()
                        results.append(result)
                        if result['status'] == 'success':
                            print(f"✓ Processed: {pdf.name}")
                        else:
                            print(f"✗ Failed: {pdf.name} - {result['error']}")
                    except Exception as e:
                        results.append({
                            'status': 'error',
                            'pdf_path': str(pdf),
                            'output_path': None,
                            'provider': self.processor.provider,
                            'model': self.processor.model,
                            'error': str(e)
                        })
                        print(f"✗ Failed: {pdf.name} - {str(e)}")
        else:
            # Process sequentially
            for pdf in pdf_files:
                result = self.process_single_pdf(pdf, output_path, prompt)
                results.append(result)
                if result['status'] == 'success':
                    print(f"✓ Processed: {pdf.name}")
                else:
                    print(f"✗ Failed: {pdf.name} - {result['error']}")
        
        # Print summary
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = len(results) - successful
        
        print(f"\nProcessing complete:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Provider: {self.processor.provider}")
        print(f"  Model: {self.processor.model}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Batch process PDF files with LLM (OpenAI or Ollama)')
    parser.add_argument('input_dir', help='Directory containing PDF files')
    parser.add_argument('--output-dir', '-o', help='Output directory for processed files')
    parser.add_argument('--prompt', '-p', help='Custom prompt for the LLM')
    parser.add_argument('--provider', choices=['openai', 'ollama'], help='LLM provider to use (openai or ollama)')
    parser.add_argument('--model', '-m', help='LLM model to use')
    parser.add_argument('--sequential', '-s', action='store_true', help='Process files sequentially (not in parallel)')
    
    args = parser.parse_args()
    
    # Validate input directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory not found: {input_path}")
        sys.exit(1)
    
    if not input_path.is_dir():
        print(f"Error: Input path is not a directory: {input_path}")
        sys.exit(1)
    
    # Set output directory
    if args.output_dir:
        output_path = Path(args.output_dir)
    else:
        output_path = input_path / "processed"
    
    try:
        # Initialize batch processor
        processor = BatchPDFProcessor(
            provider=args.provider,
            model=args.model,
            max_workers=3 if not args.sequential else 1
        )
        
        # Process directory
        results = processor.process_directory(
            input_path,
            output_path,
            args.prompt,
            parallel=not args.sequential
        )
        
        # Save processing report
        report_path = output_path / "processing_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("PDF Processing Report\n")
            f.write("=" * 50 + "\n\n")
            
            for result in results:
                f.write(f"File: {result['pdf_path']}\n")
                f.write(f"Status: {result['status']}\n")
                f.write(f"Provider: {result['provider']}\n")
                f.write(f"Model: {result['model']}\n")
                if result['output_path']:
                    f.write(f"Output: {result['output_path']}\n")
                if result['error']:
                    f.write(f"Error: {result['error']}\n")
                f.write("-" * 30 + "\n")
        
        print(f"\nProcessing report saved to: {report_path}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
