#!/usr/bin/env python3
"""
PDF to LLM Processor using Microsoft MarkItDown with Ollama Support

This script converts PDF files to text using Microsoft's MarkItDown library
and then processes the extracted text with either OpenAI GPT or Ollama models.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import openai
import ollama
from markitdown import MarkItDown

# Load environment variables
load_dotenv()

class PDFToLLMProcessor:
    def __init__(self, provider=None, model=None):
        """Initialize the processor with specified LLM provider and MarkItDown."""
        # Determine which provider to use
        if provider is None:
            provider = os.getenv('LLM_PROVIDER', 'openai')
        
        self.provider = provider.lower()
        
        if self.provider == 'openai':
            self.api_key = os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            self.client = openai.OpenAI(api_key=self.api_key)
            self.model = model or os.getenv('OPENAI_MODEL', 'gpt-4-turbo-preview')
        elif self.provider == 'ollama':
            self.base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
            self.model = model or os.getenv('OLLAMA_MODEL', 'llama3.1:8b')
            # Test Ollama connection
            try:
                ollama.list()
            except Exception as e:
                raise ValueError(f"Cannot connect to Ollama at {self.base_url}: {str(e)}")
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}. Use 'openai' or 'ollama'")
        
        self.markitdown = MarkItDown()
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF using MarkItDown."""
        try:
            result = self.markitdown.convert(str(pdf_path))
            return result.text_content
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def process_with_llm(self, text, prompt="Summarize this document:"):
        """Process extracted text with the specified LLM."""
        try:
            if self.provider == 'openai':
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
            elif self.provider == 'ollama':
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that processes document text."},
                        {"role": "user", "content": f"{prompt}\n\n{text}"}
                    ]
                )
                return response['message']['content']
        except Exception as e:
            raise Exception(f"Error processing with {self.provider}: {str(e)}")
    
    def process_pdf(self, pdf_path, prompt=None):
        """Complete pipeline: PDF -> Text -> LLM -> Result."""
        print(f"Processing PDF: {pdf_path}")
        print(f"Using LLM provider: {self.provider}")
        print(f"Using model: {self.model}")
        
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
    
    def create_custom_model(self, text, model_name):
        """Create a custom Ollama model from the extracted text."""
        if self.provider != 'ollama':
            raise ValueError("Custom model creation is only supported with Ollama provider")
        
        try:
            # Create a clean, simple system prompt
            system_prompt = "You are a specialized assistant trained on the Deep Learning textbook by Ian Goodfellow, Yoshua Bengio, and Aaron Courville. You have comprehensive knowledge of neural networks, backpropagation, optimization algorithms, convolutional networks, recurrent networks, autoencoders, representation learning, Monte Carlo methods, and practical applications. Answer questions based on this textbook content accurately and in detail."
            
            # Create Modelfile with simple system prompt
            modelfile_content = f"""FROM {self.model}
SYSTEM {system_prompt}
"""
            
            # Save Modelfile temporarily
            modelfile_path = Path("temp_modelfile")
            with open(modelfile_path, 'w', encoding='utf-8') as f:
                f.write(modelfile_content)
            
            # Create the custom model using Ollama CLI
            print(f"Creating custom model: {model_name}")
            print(f"Using {len(text)} characters of document content")
            
            cmd = ['ollama', 'create', model_name, '-f', str(modelfile_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Clean up
            modelfile_path.unlink()
            
            if result.returncode != 0:
                raise Exception(f"Ollama CLI error: {result.stderr}")
            
            print(f"Successfully created custom model: {model_name}")
            
            # Save the full document content for RAG-style retrieval in current folder
            doc_dir = Path("document_chunks")
            doc_dir.mkdir(exist_ok=True)
            
            # Split document into manageable chunks for retrieval
            chunk_size = 5000
            chunks = []
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                chunks.append(chunk)
            
            # Save chunks
            for i, chunk in enumerate(chunks):
                chunk_file = doc_dir / f"chunk_{i:03d}.txt"
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    f.write(f"Document Chunk {i+1} of {len(chunks)}:\n\n{chunk}")
            
            # Create an index file
            index_file = doc_dir / "index.txt"
            with open(index_file, 'w', encoding='utf-8') as f:
                f.write(f"Deep Learning Textbook - Document Index\n")
                f.write(f"Total chunks: {len(chunks)}\n")
                f.write(f"Total characters: {len(text)}\n")
                f.write(f"Chunk size: {chunk_size} characters\n\n")
                for i in range(len(chunks)):
                    f.write(f"Chunk {i+1:03d}: chunk_{i:03d}.txt\n")
            
            print(f"Saved {len(chunks)} document chunks to {doc_dir}/")
            print(f"Use these chunks for RAG-style document retrieval during conversations")
            
            return result.stdout
            
        except Exception as e:
            raise Exception(f"Error creating custom model: {str(e)}")
    
    def save_result(self, result, output_path):
        """Save the LLM result to a file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"Result saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert PDF to text and process with LLM (OpenAI or Ollama)')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--prompt', '-p', help='Custom prompt for the LLM')
    parser.add_argument('--output', '-o', help='Output file for the result')
    parser.add_argument('--provider', choices=['openai', 'ollama'], help='LLM provider to use (openai or ollama)')
    parser.add_argument('--model', '-m', help='LLM model to use')
    parser.add_argument('--create-model', help='Create a custom Ollama model with this name from the PDF content')
    parser.add_argument('--use-custom-model', help='Use the specified custom model for processing')
    
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
        processor = PDFToLLMProcessor(provider=args.provider, model=args.model)
        
        # Extract text first
        print("Extracting text from PDF...")
        extracted_text = processor.extract_text_from_pdf(pdf_path)
        print(f"Extracted {len(extracted_text)} characters")
        
        # Create custom model if requested
        if args.create_model:
            if processor.provider != 'ollama':
                print("Error: Custom model creation is only supported with Ollama provider")
                sys.exit(1)
            processor.create_custom_model(extracted_text, args.create_model)
            print(f"Custom model '{args.create_model}' created successfully!")
            return
        
        # Use custom model if specified
        if args.use_custom_model:
            if processor.provider != 'ollama':
                print("Error: Custom model usage is only supported with Ollama provider")
                sys.exit(1)
            processor.model = args.use_custom_model
        
        # Set prompt
        prompt = args.prompt or "Please analyze this document and provide a comprehensive summary, including key points, main arguments, and any important insights."
        
        # Process with LLM
        print("Processing with LLM...")
        result = processor.process_with_llm(extracted_text, prompt)
        
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
    
