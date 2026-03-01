# PDF to LLM Processor with Microsoft MarkItDown

A Python application that converts PDF files to text using Microsoft's MarkItDown library and processes the extracted text with Large Language Models (LLMs).

## Features

- **PDF Text Extraction**: Uses Microsoft's MarkItDown library for reliable PDF text extraction
- **LLM Integration**: Processes extracted text with OpenAI's GPT models
- **Single File Processing**: Process individual PDF files
- **Batch Processing**: Process multiple PDFs in parallel
- **Custom Prompts**: Use custom prompts for specific analysis needs
- **Flexible Output**: Save results to files or display in console

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

## Usage

### Single File Processing

Process a single PDF file:
```bash
python pdf_to_llm.py path/to/your/document.pdf
```

With custom prompt:
```bash
python pdf_to_llm.py path/to/your/document.pdf --prompt "Extract key financial metrics from this document"
```

Save output to file:
```bash
python pdf_to_llm.py path/to/your/document.pdf --output result.txt
```

Specify different model:
```bash
python pdf_to_llm.py path/to/your/document.pdf --model gpt-3.5-turbo
```

### Batch Processing

Process all PDFs in a directory:
```bash
python batch_processor.py path/to/pdfs/
```

Specify output directory:
```bash
python batch_processor.py path/to/pdfs/ --output-dir processed_results/
```

Sequential processing (disable parallel processing):
```bash
python batch_processor.py path/to/pdfs/ --sequential
```

## Environment Variables

Create a `.env` file with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4-turbo-preview
```

## Example Prompts

- **Summarization**: "Please provide a comprehensive summary of this document"
- **Key Points**: "Extract the main points and key insights from this document"
- **Financial Analysis**: "Analyze this financial document and extract key metrics"
- **Legal Review**: "Review this legal document and identify important clauses"
- **Research Analysis**: "Analyze this research paper and summarize the methodology and findings"

## Output

The processor generates:
- Text summaries and analysis from the LLM
- Processing reports for batch operations
- Structured output files for each processed PDF

## Requirements

- Python 3.7+
- OpenAI API key
- Microsoft MarkItDown library
- Internet connection for LLM processing

## Error Handling

The application includes comprehensive error handling for:
- Missing or invalid PDF files
- API connection issues
- Text extraction failures
- File system permissions

## License

This project is provided as-is for educational and development purposes.
