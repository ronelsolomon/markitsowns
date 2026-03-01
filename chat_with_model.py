#!/usr/bin/env python3
"""
Simple chat interface for interacting with custom Ollama models
"""

import os
import sys
from dotenv import load_dotenv
import ollama

# Load environment variables
load_dotenv()

def chat_with_model(model_name):
    """Interactive chat with the specified model"""
    base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    
    print(f"Connecting to Ollama at {base_url}")
    print(f"Chatting with model: {model_name}")
    print("Type 'quit' or 'exit' to end the conversation\n")
    
    # Test connection
    try:
        models = ollama.list()
        model_exists = any(model_name in model['model'] for model in models['models'])
        if not model_exists:
            print(f"Error: Model '{model_name}' not found. Available models:")
            for model in models['models']:
                print(f"  - {model['model']}")
            return
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return
    
    conversation = []
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Add user message to conversation
            conversation.append({"role": "user", "content": user_input})
            
            print("Thinking...")
            
            # Get response from model
            response = ollama.chat(
                model=model_name,
                messages=conversation
            )
            
            assistant_response = response['message']['content']
            print(f"\n{model_name}: {assistant_response}")
            
            # Add assistant response to conversation
            conversation.append({"role": "assistant", "content": assistant_response})
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python chat_with_model.py <model_name>")
        print("Example: python chat_with_model.py my-document-model")
        sys.exit(1)
    
    model_name = sys.argv[1]
    chat_with_model(model_name)

if __name__ == "__main__":
    main()
