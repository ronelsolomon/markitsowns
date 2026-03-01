#!/usr/bin/env python3
"""
PDF to LLM Processor using Microsoft MarkItDown with Ollama Support
Now includes RAG (Retrieval-Augmented Generation) for full PDF ingestion.

Dependencies:
    pip install markitdown openai ollama chromadb python-dotenv

For local embeddings (recommended):
    ollama pull nomic-embed-text
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

# RAG dependencies
try:
    import chromadb
    from chromadb.utils import embedding_functions
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# Load environment variables
load_dotenv()


# ──────────────────────────────────────────────
# Text Chunking Utilities
# ──────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Split text into overlapping chunks to preserve context across boundaries.

    Args:
        text:       Full document text.
        chunk_size: Target character length per chunk.
        overlap:    Number of characters to repeat between consecutive chunks.

    Returns:
        List of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


# ──────────────────────────────────────────────
# Main Processor Class
# ──────────────────────────────────────────────

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
            try:
                ollama.list()
            except Exception as e:
                raise ValueError(f"Cannot connect to Ollama at {self.base_url}: {str(e)}")

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}. Use 'openai' or 'ollama'")

        self.markitdown = MarkItDown()
        self._rag_collection = None  # Populated after build_rag_index()

    # ──────────────────────────────────────────
    # PDF Text Extraction
    # ──────────────────────────────────────────

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using MarkItDown."""
        try:
            result = self.markitdown.convert(str(pdf_path))
            return result.text_content
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    # ──────────────────────────────────────────
    # Standard LLM Processing
    # ──────────────────────────────────────────

    def process_with_llm(self, text: str, prompt: str = "Summarize this document:") -> str:
        """Process extracted text with the configured LLM."""
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that processes document text."},
                {"role": "user", "content": f"{prompt}\n\n{text}"}
            ]

            if self.provider == 'openai':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=2000,
                    temperature=0.7
                )
                return response.choices[0].message.content

            elif self.provider == 'ollama':
                response = ollama.chat(model=self.model, messages=messages)
                return response['message']['content']

        except Exception as e:
            raise Exception(f"Error processing with {self.provider}: {str(e)}")

    # ──────────────────────────────────────────
    # RAG: Build Index
    # ──────────────────────────────────────────

    def build_rag_index(
        self,
        text: str,
        collection_name: str = "pdf_knowledge",
        chunk_size: int = 1000,
        overlap: int = 200,
        embedding_model: str = "nomic-embed-text",
    ) -> None:
        """
        Chunk the full PDF text, embed each chunk, and store in ChromaDB.

        For OpenAI provider, uses OpenAI's text-embedding-3-small model.
        For Ollama provider, uses the specified local embedding model (default: nomic-embed-text).

        Args:
            text:             Full extracted PDF text (no size limit).
            collection_name:  ChromaDB collection name.
            chunk_size:       Characters per chunk.
            overlap:          Overlapping characters between chunks.
            embedding_model:  Embedding model name (Ollama) or ignored (OpenAI uses its own).
        """
        if not RAG_AVAILABLE:
            raise ImportError("chromadb is not installed. Run: pip install chromadb")

        print(f"Building RAG index from {len(text):,} characters...")
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        print(f"Created {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")

        # Choose embedding function based on provider
        if self.provider == 'openai':
            ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=self.api_key,
                model_name="text-embedding-3-small"
            )
        else:
            ef = embedding_functions.OllamaEmbeddingFunction(
                url=f"{self.base_url}/api/embeddings",
                model_name=embedding_model
            )

        # Set up ChromaDB (in-memory; swap to PersistentClient for disk persistence)
        client = chromadb.Client()

        # Delete existing collection with the same name if it exists
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

        self._rag_collection = client.create_collection(
            name=collection_name,
            embedding_function=ef
        )

        # Add chunks in batches to avoid overloading the embedding API
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self._rag_collection.add(
                documents=batch,
                ids=[f"chunk_{j}" for j in range(i, i + len(batch))]
            )
            print(f"  Embedded chunks {i + 1}–{i + len(batch)} of {len(chunks)}")

        print("RAG index built successfully.")

    # ──────────────────────────────────────────
    # RAG: Persist Index to Disk
    # ──────────────────────────────────────────

    def build_persistent_rag_index(
        self,
        text: str,
        persist_dir: str = "./chroma_db",
        collection_name: str = "pdf_knowledge",
        chunk_size: int = 1000,
        overlap: int = 200,
        embedding_model: str = "nomic-embed-text",
    ) -> None:
        """
        Same as build_rag_index() but persists the vector store to disk so it
        can be reloaded across sessions without re-embedding.

        Args:
            persist_dir: Directory where ChromaDB stores its data.
        """
        if not RAG_AVAILABLE:
            raise ImportError("chromadb is not installed. Run: pip install chromadb")

        print(f"Building persistent RAG index at '{persist_dir}'...")
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        print(f"Created {len(chunks)} chunks")

        if self.provider == 'openai':
            ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=self.api_key,
                model_name="text-embedding-3-small"
            )
        else:
            ef = embedding_functions.OllamaEmbeddingFunction(
                url=f"{self.base_url}/api/embeddings",
                model_name=embedding_model
            )

        client = chromadb.PersistentClient(path=persist_dir)

        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

        self._rag_collection = client.create_collection(
            name=collection_name,
            embedding_function=ef
        )

        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self._rag_collection.add(
                documents=batch,
                ids=[f"chunk_{j}" for j in range(i, i + len(batch))]
            )
            print(f"  Embedded chunks {i + 1}–{i + len(batch)} of {len(chunks)}")

        print(f"Persistent RAG index saved to '{persist_dir}'.")

    # ──────────────────────────────────────────
    # RAG: Load Existing Index from Disk
    # ──────────────────────────────────────────

    def load_rag_index(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "pdf_knowledge",
        embedding_model: str = "nomic-embed-text",
    ) -> None:
        """Load a previously persisted ChromaDB collection."""
        if not RAG_AVAILABLE:
            raise ImportError("chromadb is not installed. Run: pip install chromadb")

        if self.provider == 'openai':
            ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=self.api_key,
                model_name="text-embedding-3-small"
            )
        else:
            ef = embedding_functions.OllamaEmbeddingFunction(
                url=f"{self.base_url}/api/embeddings",
                model_name=embedding_model
            )

        client = chromadb.PersistentClient(path=persist_dir)
        self._rag_collection = client.get_collection(
            name=collection_name,
            embedding_function=ef
        )
        print(f"Loaded RAG index from '{persist_dir}' (collection: {collection_name})")

    # ──────────────────────────────────────────
    # RAG: Query
    # ──────────────────────────────────────────

    def query_rag(self, question: str, top_k: int = 5) -> str:
        """
        Retrieve the most relevant document chunks for a question and answer it.

        Args:
            question: The user's question about the PDF.
            top_k:    Number of chunks to retrieve (more = richer context, longer prompt).

        Returns:
            LLM-generated answer grounded in the retrieved chunks.
        """
        if self._rag_collection is None:
            raise RuntimeError("RAG index not built. Call build_rag_index() first.")

        # Retrieve relevant chunks
        results = self._rag_collection.query(
            query_texts=[question],
            n_results=top_k
        )
        retrieved_chunks = results['documents'][0]
        context = "\n\n---\n\n".join(retrieved_chunks)

        prompt = (
            "Answer the question below using ONLY the document excerpts provided. "
            "If the answer is not in the excerpts, say so.\n\n"
            f"Document excerpts:\n{context}\n\n"
            f"Question: {question}"
        )

        return self.process_with_llm("", prompt)

    # ──────────────────────────────────────────
    # Convenience: Full Pipeline
    # ──────────────────────────────────────────

    def process_pdf(self, pdf_path: str, prompt: str = None) -> str:
        """Standard pipeline: PDF → Text → LLM → Result (no RAG)."""
        print(f"Processing PDF: {pdf_path}")
        print(f"Using LLM provider: {self.provider} | model: {self.model}")

        extracted_text = self.extract_text_from_pdf(pdf_path)
        print(f"Extracted {len(extracted_text):,} characters")

        if prompt is None:
            prompt = (
                "Please analyze this document and provide a comprehensive summary, "
                "including key points, main arguments, and any important insights."
            )

        print("Processing with LLM...")
        return self.process_with_llm(extracted_text, prompt)

    def process_pdf_with_rag(
        self,
        pdf_path: str,
        question: str,
        chunk_size: int = 1000,
        overlap: int = 200,
        top_k: int = 5,
        persist_dir: str = None,
        embedding_model: str = "nomic-embed-text",
    ) -> str:
        """
        Full RAG pipeline: PDF → chunks → embed → retrieve → LLM answer.

        Args:
            pdf_path:        Path to the PDF.
            question:        Question to answer from the PDF.
            chunk_size:      Characters per chunk.
            overlap:         Overlap between chunks.
            top_k:           Chunks to retrieve per query.
            persist_dir:     If set, persist the index to this directory.
            embedding_model: Local embedding model name (Ollama only).

        Returns:
            LLM-generated answer.
        """
        print(f"Processing PDF with RAG: {pdf_path}")
        extracted_text = self.extract_text_from_pdf(pdf_path)
        print(f"Extracted {len(extracted_text):,} characters")

        if persist_dir:
            self.build_persistent_rag_index(
                extracted_text,
                persist_dir=persist_dir,
                chunk_size=chunk_size,
                overlap=overlap,
                embedding_model=embedding_model,
            )
        else:
            self.build_rag_index(
                extracted_text,
                chunk_size=chunk_size,
                overlap=overlap,
                embedding_model=embedding_model,
            )

        return self.query_rag(question, top_k=top_k)

    # ──────────────────────────────────────────
    # Custom Ollama Model (legacy, 50k limit)
    # ──────────────────────────────────────────

    def create_custom_model(self, text: str, model_name: str) -> str:
        """
        Create a custom Ollama model from extracted text.
        NOTE: Limited to ~50k characters. Use RAG for larger PDFs.
        """
        if self.provider != 'ollama':
            raise ValueError("Custom model creation is only supported with Ollama provider")

        MAX_CHARS = 50_000
        if len(text) > MAX_CHARS:
            print(f"Warning: Text truncated from {len(text):,} to {MAX_CHARS:,} characters.")
            print("Consider using --rag mode for full document coverage.")
            text = text[:MAX_CHARS]

        modelfile_content = (
            f"FROM {self.model}\n\n"
            "SYSTEM You are a specialized assistant trained on the following document content. "
            "Use this knowledge to answer questions accurately based on the document.\n\n"
            f"MESSAGE user Here is the document content you should know:\n{text}\n"
        )

        modelfile_path = Path("temp_modelfile")
        with open(modelfile_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)

        print(f"Creating custom Ollama model: {model_name}")
        cmd = ['ollama', 'create', model_name, '-f', str(modelfile_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        modelfile_path.unlink()

        if result.returncode != 0:
            raise Exception(f"Ollama CLI error: {result.stderr}")

        print(f"Successfully created custom model: {model_name}")
        return result.stdout

    # ──────────────────────────────────────────
    # Save Output
    # ──────────────────────────────────────────

    def save_result(self, result: str, output_path: str) -> None:
        """Save the LLM result to a file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"Result saved to: {output_path}")


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Convert PDF to text and process with LLM (OpenAI or Ollama). '
                    'Supports RAG for unlimited PDF sizes.'
    )
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--prompt', '-p', help='Custom prompt for the LLM (standard mode)')
    parser.add_argument('--question', '-q', help='Question to answer via RAG mode')
    parser.add_argument('--output', '-o', help='Output file for the result')
    parser.add_argument('--provider', choices=['openai', 'ollama'], help='LLM provider')
    parser.add_argument('--model', '-m', help='LLM model to use')

    # RAG options
    rag_group = parser.add_argument_group('RAG Options')
    rag_group.add_argument('--rag', action='store_true',
                           help='Enable RAG mode (handles PDFs of any size)')
    rag_group.add_argument('--chunk-size', type=int, default=1000,
                           help='Characters per chunk (default: 1000)')
    rag_group.add_argument('--overlap', type=int, default=200,
                           help='Overlap between chunks (default: 200)')
    rag_group.add_argument('--top-k', type=int, default=5,
                           help='Number of chunks to retrieve (default: 5)')
    rag_group.add_argument('--persist-dir', default=None,
                           help='Persist RAG index to this directory for reuse')
    rag_group.add_argument('--load-index', default=None,
                           help='Load existing RAG index from this directory (skip re-embedding)')
    rag_group.add_argument('--embedding-model', default='nomic-embed-text',
                           help='Ollama embedding model (default: nomic-embed-text)')

    # Ollama custom model options (legacy)
    ollama_group = parser.add_argument_group('Ollama Custom Model (Legacy)')
    ollama_group.add_argument('--create-model',
                              help='Create a custom Ollama model (50k char limit; prefer --rag)')
    ollama_group.add_argument('--use-custom-model',
                              help='Use a previously created custom Ollama model')

    args = parser.parse_args()

    # Validate PDF
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    if pdf_path.suffix.lower() != '.pdf':
        print(f"Error: File is not a PDF: {pdf_path}")
        sys.exit(1)

    try:
        processor = PDFToLLMProcessor(provider=args.provider, model=args.model)

        # ── RAG mode ──────────────────────────────
        if args.rag or args.question:
            question = args.question or input("Enter your question about the PDF: ")

            if args.load_index:
                # Reuse a previously persisted index
                processor.load_rag_index(
                    persist_dir=args.load_index,
                    embedding_model=args.embedding_model
                )
                result = processor.query_rag(question, top_k=args.top_k)
            else:
                result = processor.process_pdf_with_rag(
                    pdf_path=pdf_path,
                    question=question,
                    chunk_size=args.chunk_size,
                    overlap=args.overlap,
                    top_k=args.top_k,
                    persist_dir=args.persist_dir,
                    embedding_model=args.embedding_model,
                )

        # ── Legacy: create/use custom Ollama model ─
        elif args.create_model or args.use_custom_model:
            print("Extracting text from PDF...")
            extracted_text = processor.extract_text_from_pdf(pdf_path)
            print(f"Extracted {len(extracted_text):,} characters")

            if args.create_model:
                if processor.provider != 'ollama':
                    print("Error: Custom model creation requires Ollama provider")
                    sys.exit(1)
                processor.create_custom_model(extracted_text, args.create_model)
                print(f"Custom model '{args.create_model}' created. Exiting.")
                return

            if args.use_custom_model:
                if processor.provider != 'ollama':
                    print("Error: Custom model requires Ollama provider")
                    sys.exit(1)
                processor.model = args.use_custom_model

            prompt = args.prompt or (
                "Please analyze this document and provide a comprehensive summary, "
                "including key points, main arguments, and any important insights."
            )
            print("Processing with LLM...")
            result = processor.process_with_llm(extracted_text, prompt)

        # ── Standard mode ─────────────────────────
        else:
            result = processor.process_pdf(pdf_path, prompt=args.prompt)

        # ── Output ────────────────────────────────
        if args.output:
            processor.save_result(result, Path(args.output))
        else:
            print("\n" + "=" * 60)
            print("LLM PROCESSING RESULT:")
            print("=" * 60)
            print(result)
            print("=" * 60)

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()