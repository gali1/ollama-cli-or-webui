import os
import sys
import psutil  # For monitoring memory usage
import tempfile  # For temporary disk-based storage
import readline  # For command history and editing in terminal
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from llama_cpp import Llama
from dataclasses import dataclass
from typing import Optional, Generator

# Load environment variables from .env file
load_dotenv()

# Detect platform and architecture
platform = sys.platform
is_windows = platform.startswith("win")
is_linux = platform.startswith("linux")
is_macos = platform.startswith("darwin")

architecture = "unknown"
if sys.maxsize > 2**32:
    architecture = "64-bit"
else:
    architecture = "32-bit"

if sys.implementation.name == "cpython" and "arm" in platform:
    architecture += " ARM"

# Initialize Flask application
app = Flask(__name__)

# Configuration
@dataclass
class LLMConfig:
    model_name: str = os.getenv("MODEL_NAME", "gpt2")
    use_llama_cpp: bool = os.getenv("USE_LLAMA_CPP", "true").lower() in ("true", "1", "t", "y", "yes")
    llama_model_path: Optional[str] = os.getenv("LLAMA_MODEL_PATH")
    max_tokens: int = int(os.getenv("MAX_TOKENS", "256"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    top_p: float = float(os.getenv("TOP_P", "0.95"))
    top_k: int = int(os.getenv("TOP_K", "50"))
    repetition_penalty: float = float(os.getenv("REPETITION_PENALTY", "1.2"))
    no_repeat_ngram_size: int = int(os.getenv("NO_REPEAT_NGRAM_SIZE", "4"))
    num_beams: int = int(os.getenv("NUM_BEAMS", "1"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "128"))

config = LLMConfig()

# Load model and tokenizer
if config.use_llama_cpp:
    if not config.llama_model_path:
        raise ValueError("LLAMA_MODEL_PATH must be set when USE_LLAMA_CPP is True")
    model = Llama(model_path=config.llama_model_path)
    tokenizer = None
else:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

# Helper functions
def check_memory_usage() -> bool:
    """Check current memory usage and return True if high."""
    memory_info = psutil.virtual_memory()
    return memory_info.percent > 80  # Threshold of 80% usage

def generate_response_transformers(prompt: str) -> str:
    """Generate response using the Transformers model."""
    if tokenizer is None:
        raise RuntimeError("Tokenizer is not available for Transformers model.")

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=input_ids.shape[1] + config.max_tokens,
        num_return_sequences=1,
        no_repeat_ngram_size=config.no_repeat_ngram_size,
        do_sample=True,
        top_k=config.top_k,
        top_p=config.top_p,
        temperature=config.temperature,
        repetition_penalty=config.repetition_penalty,
        num_beams=config.num_beams,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(output[0], skip_special_tokens=True).strip()

def generate_response_llama(prompt: str) -> Generator[str, None, None]:
    """Generate response using the Llama model."""
    try:
        response = model.create_completion(prompt, max_tokens=config.max_tokens)

        if isinstance(response, dict):
            choices = response.get('choices', [])
            if choices and isinstance(choices[0], dict):
                text = choices[0].get('text', '')
            else:
                text = ''
        else:
            text = response

        formatted_text = text.strip().replace('\n', '\n\n')  # Double newlines for readability

        for chunk in formatted_text.split('\n\n'):
            yield chunk
    except Exception as e:
        print(f"Error in Llama model generation: {str(e)}")
        yield "Error generating response."

@app.route("/generate", methods=["POST"])
def generate():
    """Handle POST requests to generate responses."""
    data = request.json
    prompt = data.get("prompt", "")

    if check_memory_usage():
        return jsonify({"warning": "High memory usage detected. Disk-based storage is in effect."}), 503

    try:
        if config.use_llama_cpp:
            response_generator = generate_response_llama(prompt)
        else:
            response_generator = generate_response_transformers(prompt)

        response_text = next(response_generator)
        return jsonify({"response": response_text})
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return jsonify({"error": "Error generating response"}), 500

if __name__ == "__main__":
    # Enable command history and editing with readline
    readline.parse_and_bind('tab: complete')
    readline.parse_and_bind('set editing-mode vi')  # Optional: Use vi-style editing

    history_file = os.path.join(tempfile.gettempdir(), "query_history")
    if os.path.exists(history_file):
        readline.read_history_file(history_file)

    print("Enter your prompt (type 'exit' to quit):")

    try:
        while True:
            prompt = input("> ")
            if prompt.lower() == 'exit':
                break

            if prompt:
                if check_memory_usage():
                    print("High memory usage detected. Using disk-based storage.")
                    # Use temporary files for storage or similar strategy here if necessary
                if config.use_llama_cpp:
                    response = generate_response_llama(prompt)
                else:
                    response = generate_response_transformers(prompt)
                
                for chunk in response:
                    print(chunk)

                # Save the prompt to history file
                readline.add_history(prompt)
                readline.write_history_file(history_file)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nExiting...")
